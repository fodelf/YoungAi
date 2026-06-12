#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_distributed.h"   /* remote expert pread client (P2.2 low-cost variant) */

/*
 * Objective-C Metal glue for the C engine.
 *
 * The C code owns model semantics and graph scheduling.  This file owns only
 * Metal objects: device/queue/library setup, mmap-backed weight views, command
 * batching, persistent tensors, scratch buffers, and thin wrappers around the
 * kernel files in the metal directory.  Keeping this boundary narrow makes the
 * inference path readable from C while still using Objective-C where Metal
 * requires it.
 */

enum {
    DS4_METAL_TENSOR_Q2_K    = 10,
    DS4_METAL_TENSOR_Q4_K    = 12,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
};

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
static id<MTLComputePipelineState> g_set_rows_f32_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f16_pipeline;
static id<MTLComputePipelineState> g_get_rows_i32_pipeline;
static id<MTLComputePipelineState> g_repeat_f32_pipeline;
static id<MTLComputePipelineState> g_concat_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f32_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_add_pipeline;
static id<MTLComputePipelineState> g_moe_sum6_pipeline;
static id<MTLComputePipelineState> g_mul_pipeline;
static id<MTLComputePipelineState> g_rms_norm_pipeline;
static id<MTLComputePipelineState> g_rms_norm_plain_pipeline;
static id<MTLComputePipelineState> g_dsv4_qkv_rms_norm_pipeline;
static id<MTLComputePipelineState> g_hc_split_sinkhorn_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_norm_pipeline;
static id<MTLComputePipelineState> g_hc_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_expand_pipeline;
static id<MTLComputePipelineState> g_unary_sigmoid_pipeline;
static id<MTLComputePipelineState> g_unary_silu_pipeline;
static id<MTLComputePipelineState> g_unary_softplus_pipeline;
static id<MTLComputePipelineState> g_unary_sqrt_pipeline;
static id<MTLComputePipelineState> g_unary_clamp_pipeline;
static id<MTLComputePipelineState> g_unary_scale_pipeline;
static id<MTLComputePipelineState> g_unary_fill_pipeline;
static id<MTLComputePipelineState> g_unary_fill_f16_pipeline;
static id<MTLComputePipelineState> g_bin_mul_scalar_pipeline;
static id<MTLComputePipelineState> g_bin_div_row_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_rope_tail_batch_pipeline;
static id<MTLComputePipelineState> g_dsv4_fp8_kv_quantize_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_qat_pipeline;
static id<MTLComputePipelineState> g_dsv4_kv_fp8_store_pipeline;
static id<MTLComputePipelineState> g_dsv4_ratio4_shift_pipeline;
static id<MTLComputePipelineState> g_dsv4_softmax_pool_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_4_pipeline;
static id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_sum_rows_f32_f32_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_scatter_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_score_one_direct_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_store_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_sort_i32_rows_asc_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_rb16_pipeline;
static id<MTLComputePipelineState> g_dsv4_softplus_sqrt_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_route_translate_pipeline;
static id<MTLComputePipelineState> g_dsv4_hc_expand4_pipeline;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;
static NSMutableArray<id<MTLBuffer>> *g_transient_buffers;
static id g_model_residency_set;
static id<MTLBuffer> g_flash_attn_mask_buffer;
static id<MTLBuffer> g_flash_attn_pad_buffer;
static id<MTLBuffer> g_flash_attn_tmp_buffer;
static id<MTLBuffer> g_flash_attn_blk_buffer;
static id<MTLBuffer> g_flash_attn_ring_buffer;
static id<MTLBuffer> g_flash_attn_kv_buffer;
static id<MTLBuffer> g_compressor_pool_kv_buffer;
static id<MTLBuffer> g_compressor_pool_score_buffer;
static id<MTLBuffer> g_compressor_pool_score_cont_buffer;
static id<MTLBuffer> g_compressor_pool_softmax_buffer;
static id<MTLBuffer> g_compressor_pool_product_buffer;
static id<MTLBuffer> g_compressor_store_ape_buffer;
static id<MTLBuffer> g_compressor_store_score_buffer;
static id<MTLBuffer> g_embed_rows_buffer;
static id<MTLBuffer> g_router_selection_buffer;
static id<MTLBuffer> g_router_weight_sum_buffer;
/* Resident original-id -> compact-slot LUT for reduced-expert models, n_layer*256
 * int16. nil for a full model (translation kernel never dispatched). */
static id<MTLBuffer> g_expert_keep_lut_buffer;
static uint32_t g_expert_keep_lut_layers;
static id<MTLBuffer> g_indexer_head_scores_buffer;
static id<MTLBuffer> g_indexer_topk_buffer;
static id<MTLBuffer> g_indexed_topk_buffer;
static id<MTLBuffer> g_f16_round_scratch_buffer;
static id<MTLBuffer> g_raw_store_round_buffer;
static id<MTLBuffer> g_moe_gate_scratch_buffer;
static id<MTLBuffer> g_moe_down_scratch_buffer;
static id<MTLBuffer> g_moe_id_map_buffer;
static id<MTLBuffer> g_attn_out_group_ids_buffer;
/* A3 routed-expert offload scratch.  When DS4_METAL_EXPERT_OFFLOAD=1 the
 * routed MoE kernels read only the active expert slots copied from the GGUF mmap
 * into these compact resident buffers, instead of binding the huge mmap-backed
 * expert tensors to each command buffer. */
static id<MTLBuffer> g_moe_scratch_gate;
static id<MTLBuffer> g_moe_scratch_up;
static id<MTLBuffer> g_moe_scratch_down;
static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_mapped_offset;
static uint64_t g_model_mapped_size;
static uint64_t g_model_mapped_max_tensor_bytes;
static uint64_t g_tensor_alloc_live_bytes;
static uint64_t g_tensor_alloc_peak_bytes;
static uint64_t g_model_wrap_count;
static uint64_t g_model_wrap_bytes;
static uint64_t g_model_wrap_max_bytes;
static uint64_t g_model_residency_count;
static int g_metal4_runtime_available;
static int g_metal4_family_supported;
static int g_metal4_queue_supported;
static int g_metal4_m5_neural_accelerators_hint;
static int g_metal4_tensor_api_enabled;
static int g_metal4_tensor_api_compile_supported;
static char g_metal_device_name[128];
static NSUInteger g_flash_attn_mask_bytes;
static NSUInteger g_flash_attn_pad_bytes;
static NSUInteger g_flash_attn_tmp_bytes;
static NSUInteger g_flash_attn_blk_bytes;
static NSUInteger g_flash_attn_ring_bytes;
static NSUInteger g_flash_attn_kv_bytes;
static NSUInteger g_compressor_pool_kv_bytes;
static NSUInteger g_compressor_pool_score_bytes;
static NSUInteger g_compressor_pool_score_cont_bytes;
static NSUInteger g_compressor_pool_softmax_bytes;
static NSUInteger g_compressor_pool_product_bytes;
static NSUInteger g_compressor_store_ape_bytes;
static NSUInteger g_compressor_store_score_bytes;
static NSUInteger g_embed_rows_bytes;
static NSUInteger g_router_selection_bytes;
static NSUInteger g_router_weight_sum_bytes;
static NSUInteger g_indexer_head_scores_bytes;
static NSUInteger g_indexer_topk_bytes;
static NSUInteger g_indexed_topk_bytes;
static NSUInteger g_f16_round_scratch_bytes;
static NSUInteger g_raw_store_round_bytes;
static NSUInteger g_moe_gate_scratch_bytes;
static NSUInteger g_moe_down_scratch_bytes;
static NSUInteger g_moe_id_map_bytes;
static NSUInteger g_attn_out_group_ids_bytes;
static NSUInteger g_moe_scratch_gate_bytes;
static NSUInteger g_moe_scratch_up_bytes;
static NSUInteger g_moe_scratch_down_bytes;
static int g_initialized;
static int g_quality_mode;
static int g_mpp_invalid_env_reported;

static uint64_t ds4_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void ds4_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = ds4_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "ds4: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "ds4: Metal device %s\n", name);
    }
}

#define DS4_METAL_MAX_MODEL_VIEWS 4096
/* Compatibility fallback for callers that cannot provide a parsed GGUF tensor
 * span. The normal DS4 engine passes the exact maximum tensor byte size. */
#define DS4_METAL_FALLBACK_MAX_TENSOR_BYTES (4ull * 1024ull * 1024ull * 1024ull)

typedef struct {
    __strong id<MTLBuffer> buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t model_offset;
    uint64_t bytes;
    /* When false, this view is wrapped (so hot-path ds4_gpu_wrap_model_range can
     * still resolve a buffer for any tensor it covers) but is NOT added to the
     * model MTLResidencySet. Its file-backed clean pages stay reclaimable under
     * memory pressure. Used for routed-expert weights under DS4_METAL_EXPERT_OFFLOAD;
     * every view is resident by default (full-model behaviour unchanged). */
    bool resident_hint;
} ds4_gpu_model_view;

static ds4_gpu_model_view g_model_views[DS4_METAL_MAX_MODEL_VIEWS];
static uint32_t g_model_view_count;

@interface DS4MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
@end

@implementation DS4MetalTensor
@end

static DS4MetalTensor *ds4_gpu_tensor_obj(ds4_gpu_tensor *tensor) {
    return (__bridge DS4MetalTensor *)tensor;
}

static const DS4MetalTensor *ds4_gpu_tensor_const_obj(const ds4_gpu_tensor *tensor) {
    return (__bridge const DS4MetalTensor *)tensor;
}

static id<MTLBuffer> ds4_gpu_tensor_buffer(const ds4_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static NSUInteger ds4_gpu_tensor_offset(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static id<MTLCommandBuffer> ds4_gpu_command_buffer(int *owned) {
    if (g_batch_cb) {
        *owned = 0;
        return g_batch_cb;
    }
    *owned = 1;
    return [g_queue commandBuffer];
}

static id<MTLComputeCommandEncoder> ds4_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        if (!g_batch_enc) g_batch_enc = [cb computeCommandEncoder];
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static void ds4_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static void ds4_gpu_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

static int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
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

static int ds4_gpu_wait_pending_command_buffers(const char *label) {
    int ok = 1;
    for (id<MTLCommandBuffer> pending in g_pending_cbs) {
        if (!ds4_gpu_wait_command_buffer(pending, label)) ok = 0;
    }
    [g_pending_cbs removeAllObjects];
    return ok;
}

static int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    if (!owned) return 1;

    [cb commit];
    int ok = ds4_gpu_wait_pending_command_buffers(label);
    if (!ds4_gpu_wait_command_buffer(cb, label)) ok = 0;
    [g_transient_buffers removeAllObjects];
    return ok;
}

static int ds4_gpu_device_name_contains(const char *needle);

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

static int ds4_gpu_ensure_scratch_buffer(
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

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static uint64_t ds4_gpu_effective_model_max_tensor_bytes(uint64_t map_size, uint64_t max_tensor_bytes) {
    if (max_tensor_bytes != 0) return max_tensor_bytes;
    return map_size < DS4_METAL_FALLBACK_MAX_TENSOR_BYTES ?
           map_size : DS4_METAL_FALLBACK_MAX_TENSOR_BYTES;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(const char *function_name);
static int ds4_gpu_warm_model_views(void);
static double ds4_gpu_gib(uint64_t bytes);

static double ds4_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int ds4_gpu_progress_enabled(void) {
    return ds4_log_is_tty(stderr);
}

static void ds4_gpu_progress_begin(const char *what) {
    if (!ds4_gpu_progress_enabled()) return;
    fprintf(stderr, "ds4: %s...", what);
    fflush(stderr);
}

static void ds4_gpu_progress_done(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_progress_failed(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_model_views_clear(void) {
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        g_model_views[i].buffer = nil;
        g_model_views[i].model_map = NULL;
        g_model_views[i].model_size = 0;
        g_model_views[i].model_offset = 0;
        g_model_views[i].bytes = 0;
    }
    g_model_view_count = 0;
}

static void ds4_gpu_model_residency_clear(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_model_residency_set) {
            [g_model_residency_set endResidency];
            [g_model_residency_set removeAllAllocations];
            g_model_residency_set = nil;
        }
    }
#endif
    g_model_residency_count = 0;
}

static int ds4_gpu_model_residency_request_views(void) {
    if (g_model_view_count == 0 || getenv("DS4_METAL_NO_RESIDENCY") != NULL) return 1;

#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        /*
         * Register all model views as one residency set before inference. This
         * is a GPU residency/budgeting hint, not a request to fault the whole
         * 80+ GB file into memory. Its purpose is to make the driver see the
         * complete set of large shared allocations during setup instead of
         * discovering them lazily from the first measured graph command, where
         * VM validation and residency accounting would look like model compute.
         */
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_model";
        desc.initialCapacity = g_model_view_count;

        NSError *error = nil;
        g_model_residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!g_model_residency_set) {
            fprintf(stderr, "ds4: Metal model residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        uint32_t resident_views = 0;
        uint64_t resident_bytes = 0;
        for (uint32_t i = 0; i < g_model_view_count; i++) {
            /* Non-resident views (routed-expert offload) are wrapped but kept out
             * of the residency set so their clean pages stay reclaimable. */
            if (!g_model_views[i].resident_hint) continue;
            [g_model_residency_set addAllocation:g_model_views[i].buffer];
            resident_views++;
            resident_bytes += g_model_views[i].bytes;
        }
        [g_model_residency_set commit];
        [g_model_residency_set requestResidency];
        g_model_residency_count = resident_views;
        /* [diag] surface the actual wired working set vs the GPU ceiling. This
         * fires once per model map (base slice, then again after MTP appends its
         * views), so the second line shows the cumulative base+MTP residency. */
        fprintf(stderr,
                "ds4: [diag] residency set: %u/%u views wired, %.2f GiB; "
                "device recommendedMax %.2f GiB, currentAllocated %.2f GiB\n",
                resident_views, g_model_view_count,
                (double)resident_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)[g_device recommendedMaxWorkingSetSize] / (1024.0 * 1024.0 * 1024.0),
                (double)[g_device currentAllocatedSize] / (1024.0 * 1024.0 * 1024.0));
    }
#endif

    return 1;
}

static int ds4_gpu_add_model_view_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes,
        bool        resident,
        uint64_t   *mapped_model_size_out) {
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)model_map;

    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "ds4: Metal model mmap base is not page aligned\n");
        return 0;
    }
    if (map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "ds4: Metal model mapped range is outside the GGUF mapping\n");
        return 0;
    }
    const uint64_t page_model_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_model_offset;
    if (map_size > UINT64_MAX - leading ||
        leading + map_size > UINT64_MAX - (page - 1))
    {
        fprintf(stderr, "ds4: Metal model mapped range overflows page alignment\n");
        return 0;
    }
    const uint64_t mapped_model_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
    max_buffer &= ~(page - 1);

    /*
     * Wrap only the tensor-data part of the GGUF file. Metadata is parsed by the
     * CPU and is never dereferenced by kernels, so exposing it to Metal only
     * grows the residency set and the VM range the driver must validate.
     *
     * Metal buffers have a device-specific maximum length, and this model is
     * larger than that maximum on the target machines. Creating one no-copy
     * buffer per tensor would avoid the length limit, but it would also move a
     * lot of VM-object creation and residency bookkeeping into graph setup. The
     * stable shape here is a tiny number of page-aligned views created once.
     *
     * Adjacent views intentionally overlap by more than the largest tensor, plus
     * one page for alignment. That invariant guarantees every tensor lies wholly
     * inside at least one view, so hot paths pass one buffer and one inner byte
     * offset. We never split a weight tensor across command encoders.
     */
    if (max_tensor_bytes > map_size) {
        fprintf(stderr, "ds4: Metal model max tensor span is larger than a mapped tensor span\n");
        return 0;
    }
    if (max_tensor_bytes > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "ds4: Metal model max tensor span overflows page alignment\n");
        return 0;
    }
    const uint64_t max_tensor_rounded = round_up_u64(max_tensor_bytes, page);
    if (max_tensor_rounded > UINT64_MAX - page) {
        fprintf(stderr, "ds4: Metal model view overlap overflows page slack\n");
        return 0;
    }
    const uint64_t overlap = max_tensor_rounded + page;
    if (max_buffer == 0 || max_buffer <= overlap) {
        fprintf(stderr,
                "ds4: Metal maxBufferLength is too small for DS4 model views "
                "(max tensor %.2f GiB, max buffer %.2f GiB)\n",
                ds4_gpu_gib(max_tensor_bytes),
                ds4_gpu_gib(max_buffer));
        return 0;
    }

    const uint64_t step = max_buffer - overlap;
    uint64_t off = 0;
    while (off < mapped_model_size) {
        if (g_model_view_count == DS4_METAL_MAX_MODEL_VIEWS) {
            fprintf(stderr, "ds4: Metal model needs more mapped views than expected\n");
            return 0;
        }

        uint64_t view_bytes = mapped_model_size - off;
        if (view_bytes > max_buffer) view_bytes = max_buffer;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                                           length:(NSUInteger)view_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n",
                    (double)(page_model_offset + off) / (1024.0 * 1024.0 * 1024.0),
                    (double)view_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"ds4_model_view_%u", g_model_view_count];

        g_model_views[g_model_view_count].buffer = buffer;
        g_model_views[g_model_view_count].model_map = model_map;
        g_model_views[g_model_view_count].model_size = model_size;
        g_model_views[g_model_view_count].model_offset = page_model_offset + off;
        g_model_views[g_model_view_count].bytes = view_bytes;
        g_model_views[g_model_view_count].resident_hint = resident;
        g_model_view_count++;

        g_model_wrap_count++;
        g_model_wrap_bytes += view_bytes;
        if (view_bytes > g_model_wrap_max_bytes) g_model_wrap_max_bytes = view_bytes;

        if (off + view_bytes >= mapped_model_size) break;
        off += step;
    }

    if (mapped_model_size_out) *mapped_model_size_out += mapped_model_size;
    return 1;
}

static int ds4_gpu_finish_model_views(
        double t0,
        uint64_t mapped_model_size,
        uint64_t display_offset) {
    const double t_mapped = ds4_gpu_now_ms();
    const int request_residency = getenv("DS4_METAL_NO_RESIDENCY") == NULL;
    if (request_residency) ds4_gpu_progress_begin("requesting Metal residency (may take tens of seconds)");
    if (!ds4_gpu_model_residency_request_views()) {
        if (request_residency) ds4_gpu_progress_failed();
        return 0;
    }
    if (request_residency) ds4_gpu_progress_done();
    const double t_resident = ds4_gpu_now_ms();
    int warmed = 1;
    const double t_warm0 = ds4_gpu_now_ms();
    const int warm_model_views = getenv("DS4_METAL_NO_RESIDENCY") == NULL &&
                                 getenv("DS4_METAL_NO_MODEL_WARMUP") == NULL;
    if (warm_model_views) {
        /*
         * The first GPU command touching no-copy mmap storage can pay command
         * queue setup, page-table validation, and shared-allocation residency
         * costs. Sample each model view here so timed graph execution starts
         * after that one-time work. The stride is intentionally coarse: this is
         * a validation touch over the VM ranges, not a full model prefetch. A
         * dense prefetch would create exactly the kind of memory pressure and
         * startup stalls this path is designed to avoid.
         */
        ds4_gpu_progress_begin("warming Metal model views");
        warmed = ds4_gpu_warm_model_views();
        if (warmed) ds4_gpu_progress_done();
        else ds4_gpu_progress_failed();
    }
    const double t_warm = ds4_gpu_now_ms();
    fprintf(stderr,
            "ds4: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n",
            t_mapped - t0,
            t_resident - t_mapped,
            t_warm - t_warm0,
            mapped_model_size / 1024.0 / 1024.0,
            display_offset / 1024.0 / 1024.0);
    if (!warmed) return 0;
    return 1;
}

static int ds4_gpu_map_model_views(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size,
        uint64_t    max_tensor_bytes) {
    const double t0 = ds4_gpu_now_ms();
    uint64_t mapped_model_size = 0;
    if (!ds4_gpu_add_model_view_range(model_map,
                                      model_size,
                                      map_offset,
                                      map_size,
                                      max_tensor_bytes,
                                      true,
                                      &mapped_model_size)) {
        return 0;
    }
    return ds4_gpu_finish_model_views(t0, mapped_model_size, map_offset);
}

static id<MTLBuffer> ds4_gpu_new_transient_buffer(NSUInteger bytes, const char *label) {
    if (bytes == 0) bytes = 1;

    id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal transient buffer %s (%llu bytes)\n",
                label ? label : "(unnamed)", (unsigned long long)bytes);
        return nil;
    }
    if (label) buffer.label = [NSString stringWithUTF8String:label];

    /*
     * CPU-filled buffers must survive until their command buffer completes.
     * A local ObjC strong variable is not enough when the encoder function
     * returns before the caller commits the command buffer.
     */
    [g_transient_buffers addObject:buffer];
    return buffer;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d",
                     function_name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_id_pipeline(
        const char *function_name,
        bool        bc_inp) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d",
                     function_name, bc_inp ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found\n", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static int ds4_gpu_disable_hot_pipeline_statics(void) {
    static int initialized;
    static int disabled;
    if (!initialized) {
        disabled = getenv("DS4_METAL_DISABLE_HOT_PIPELINE_STATICS") != NULL;
        initialized = 1;
    }
    return disabled;
}

static id<MTLComputePipelineState> ds4_gpu_hot_pipeline(
        id<MTLComputePipelineState> pipeline,
        const char *fallback_name) {
    if (!ds4_gpu_disable_hot_pipeline_statics()) return pipeline;
    return ds4_gpu_get_pipeline(fallback_name);
}

static int ds4_gpu_use_compressor_pair_nr4(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_COMPRESSOR_PAIR_NR4") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_device_name_contains(const char *needle);

static int ds4_gpu_env_value_eq(const char *v, size_t n, const char *literal) {
    size_t m = strlen(literal);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) != tolower((unsigned char)literal[i])) return 0;
    }
    return 1;
}

static int ds4_gpu_env_bool(const char *name) {
    const char *v = getenv(name);
    if (!v) return -1;

    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (n == 0) return 1;

    if (ds4_gpu_env_value_eq(v, n, "1") ||
        ds4_gpu_env_value_eq(v, n, "true") ||
        ds4_gpu_env_value_eq(v, n, "yes") ||
        ds4_gpu_env_value_eq(v, n, "on")) {
        return 1;
    }
    if (ds4_gpu_env_value_eq(v, n, "0") ||
        ds4_gpu_env_value_eq(v, n, "false") ||
        ds4_gpu_env_value_eq(v, n, "no") ||
        ds4_gpu_env_value_eq(v, n, "off")) {
        return 0;
    }

    if (!g_mpp_invalid_env_reported) {
        fprintf(stderr,
                "ds4: invalid Metal boolean environment value %s=%.*s; treating presence as enabled\n",
                name, (int)n, v);
        g_mpp_invalid_env_reported = 1;
    }
    return 1;
}

static int ds4_gpu_mpp_available(void) {
    return g_metal4_tensor_api_enabled && !g_quality_mode;
}

/*
 * Retained Metal4 defaults live here instead of behind user-visible options.
 * The public runtime has one automatic accelerated path plus the global
 * DS4_METAL_DISABLE_METAL4 comparison switch.  Benchmark-only alternatives that
 * lost during M5 work are removed or kept out of the dispatch path so future
 * changes do not accidentally turn old experiments into new modes.
 */
static int ds4_gpu_use_mpp_attn_out_low_matmul(void) {
    return ds4_gpu_mpp_available();
}

enum {
    DS4_METAL_ATTN_OUT_MPP_TILE_N = 64,
};

static void ds4_gpu_warn_mpp_fallback(void) {
    static int warned;
    if (!warned) {
        fprintf(stderr, "ds4: accelerated Metal prefill matmul unavailable; falling back to legacy kernel\n");
        warned = 1;
    }
}

static int ds4_gpu_device_name_contains(const char *needle) {
    return g_metal_device_name[0] != '\0' && strstr(g_metal_device_name, needle) != NULL;
}

static int ds4_gpu_compile_tensor_probe(void) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (!g_device) return 0;
    if (@available(macOS 26.0, *)) {
        const char *src =
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void ds4_tensor_probe(\n"
            "        tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],\n"
            "        tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],\n"
            "        device float *C [[buffer(2)]],\n"
            "        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
            "    auto tA = A.slice(0, (int)tgid.y);\n"
            "    auto tB = B.slice((int)tgid.x, 0);\n"
            "    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n"
            "    auto sA = tA.slice(0, 0);\n"
            "    auto sB = tB.slice(0, 0);\n"
            "    mm.run(sB, sA, cT);\n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));\n"
            "    cT.store(tC);\n"
            "}\n";

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:src];
        id<MTLLibrary> probe_library = [g_device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!probe_library) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        id<MTLFunction> fn = [probe_library newFunctionWithName:@"ds4_tensor_probe"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe function missing\n");
            return 0;
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe pipeline failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        return 1;
    }
#endif
    return 0;
}

static void ds4_gpu_detect_metal4_features(void) {
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;
    g_metal_device_name[0] = '\0';

    if (!g_device) return;

    const char *name = [[g_device name] UTF8String];
    if (name) {
        snprintf(g_metal_device_name, sizeof(g_metal_device_name), "%s", name);
    }

    const int metal4_disabled = ds4_gpu_env_bool("DS4_METAL_DISABLE_METAL4") > 0;

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        g_metal4_runtime_available = 1;
        g_metal4_family_supported =
            !metal4_disabled && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        g_metal4_queue_supported = [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;

        /*
         * Apple does not currently expose a separate "Neural Accelerator" bit
         * through Metal. On public M5 systems the hardware signal is the device
         * generation plus Metal 4 support, so keep this as a conservative hint.
         */
        if (g_metal4_family_supported && ds4_gpu_device_name_contains("M5")) {
            g_metal4_m5_neural_accelerators_hint = 1;
        }

        if (g_metal4_family_supported) {
            const int default_enable =
                ds4_gpu_device_name_contains("M5") ||
                ds4_gpu_device_name_contains("M6") ||
                ds4_gpu_device_name_contains("A19") ||
                ds4_gpu_device_name_contains("A20");

            /*
             * Metal 4 TensorOps are portable in source, but on pre-M5 hardware
             * they can map to ordinary shader fallbacks.  Keep the automatic
             * fast path restricted to hardware generations where the Neural
             * Accelerator/TensorOps path is expected to pay off; older Metal
             * machines continue to use the established kernels unless a future
             * device is explicitly added here.
             */
            if (default_enable) {
                g_metal4_tensor_api_compile_supported = ds4_gpu_compile_tensor_probe();
                g_metal4_tensor_api_enabled = g_metal4_tensor_api_compile_supported;
                if (!g_metal4_tensor_api_enabled) {
                    fprintf(stderr, "ds4: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
                }
            } else {
                fprintf(stderr, "ds4: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
            }
        }
    }
#endif
}

static int ds4_gpu_warm_model_views(void) {
    if (g_model_view_count == 0) return 1;

    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_touch_u8_stride");
    if (!pipeline) return 0;

    uint64_t stride = 1024ull * 1024ull;
    const char *stride_env = getenv("DS4_METAL_MODEL_WARMUP_STRIDE_MB");
    if (stride_env && stride_env[0]) {
        char *end = NULL;
        unsigned long long mb = strtoull(stride_env, &end, 10);
        if (end != stride_env && mb > 0 && mb <= 1024) {
            stride = mb * 1024ull * 1024ull;
        }
    }

    uint64_t total_touches = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (!g_model_views[i].resident_hint) continue;
        total_touches += (g_model_views[i].bytes + stride - 1) / stride;
    }
    if (total_touches == 0) return 1;
    if (total_touches > (uint64_t)NSUIntegerMax) return 0;

    const NSUInteger out_bytes = (NSUInteger)total_touches;
    id<MTLBuffer> out = [g_device newBufferWithLength:out_bytes
                                             options:MTLResourceStorageModeShared];
    if (!out) {
        fprintf(stderr, "ds4: Metal model warmup scratch allocation failed\n");
        return 0;
    }
    out.label = @"ds4_model_warmup";

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) {
        fprintf(stderr, "ds4: Metal model warmup command buffer allocation failed\n");
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    uint64_t dst_offset = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (!g_model_views[i].resident_hint) continue;
        const uint64_t bytes = g_model_views[i].bytes;
        const uint64_t n = (bytes + stride - 1) / stride;
        [enc setBuffer:g_model_views[i].buffer offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
        [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dst_offset += n;
    }
    ds4_gpu_end_compute_encoder(cb, enc);

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal model warmup failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return 1;
}

static const char *ds4_gpu_mul_mm_id_map0_name(uint32_t ne20) {
    switch (ne20) {
        case 1:  return "kernel_mul_mm_id_map0_ne20_1";
        case 2:  return "kernel_mul_mm_id_map0_ne20_2";
        case 4:  return "kernel_mul_mm_id_map0_ne20_4";
        case 5:  return "kernel_mul_mm_id_map0_ne20_5";
        case 6:  return "kernel_mul_mm_id_map0_ne20_6";
        case 8:  return "kernel_mul_mm_id_map0_ne20_8";
        case 10: return "kernel_mul_mm_id_map0_ne20_10";
        case 16: return "kernel_mul_mm_id_map0_ne20_16";
        case 22: return "kernel_mul_mm_id_map0_ne20_22";
        default: return NULL;
    }
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_pipeline(
        const char *function_name,
        int16_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d", function_name, (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_ext_pipeline(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d_nxpsg=%d",
                     function_name, (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg   type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pad_pipeline(
        bool    has_mask,
        int32_t ncpsg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_pad_mask=%d_ncpsg=%d",
                     has_mask ? 1 : 0, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:100];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_blk_pipeline(
        int32_t nqptg,
        int32_t ncpsg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_blk_nqptg=%d_ncpsg=%d",
                     (int)nqptg, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:225];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        bc_mask,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     bc_mask ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:300];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:301];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:302];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:303];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:304];
    [constants setConstantValue:&bc_mask   type:MTLDataTypeBool atIndex:310];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:320];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:321];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:322];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_vec_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg,
                     (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_pipeline(
        int32_t dv,
        int32_t nwg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_vec_reduce_dv=%d_nwg=%d",
                     (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv  type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static uint32_t ds4_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg) {
    uint32_t nsg = 1;
    while (2u * nwg * nsg * ncpsg < n_keys && nsg < 4u) {
        nsg *= 2u;
    }
    return nsg;
}

static int ds4_gpu_trace_allocs(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_TRACE_ALLOCS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static double ds4_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static double ds4_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

void ds4_gpu_print_memory_report(const char *label) {
    const uint64_t scratch =
        (uint64_t)g_flash_attn_mask_bytes +
        (uint64_t)g_flash_attn_pad_bytes +
        (uint64_t)g_flash_attn_tmp_bytes +
        (uint64_t)g_flash_attn_blk_bytes +
        (uint64_t)g_flash_attn_ring_bytes +
        (uint64_t)g_flash_attn_kv_bytes +
        (uint64_t)g_compressor_pool_kv_bytes +
        (uint64_t)g_compressor_pool_score_bytes +
        (uint64_t)g_compressor_pool_score_cont_bytes +
        (uint64_t)g_compressor_pool_softmax_bytes +
        (uint64_t)g_compressor_pool_product_bytes +
        (uint64_t)g_compressor_store_ape_bytes +
        (uint64_t)g_compressor_store_score_bytes +
        (uint64_t)g_embed_rows_bytes +
        (uint64_t)g_router_selection_bytes +
        (uint64_t)g_router_weight_sum_bytes +
        (uint64_t)g_indexer_head_scores_bytes +
        (uint64_t)g_indexer_topk_bytes +
        (uint64_t)g_indexed_topk_bytes +
        (uint64_t)g_f16_round_scratch_bytes +
        (uint64_t)g_raw_store_round_bytes +
        (uint64_t)g_moe_gate_scratch_bytes +
        (uint64_t)g_moe_down_scratch_bytes +
        (uint64_t)g_moe_id_map_bytes;

    fprintf(stderr, "ds4: Metal memory report%s%s\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "");
    fprintf(stderr,
            "ds4:   runtime tensors live %.2f MiB peak %.2f MiB\n",
            ds4_gpu_mib(g_tensor_alloc_live_bytes),
            ds4_gpu_mib(g_tensor_alloc_peak_bytes));
    fprintf(stderr,
            "ds4:   mmap model wrapper spans %llu buffers %.2f GiB total, %.2f GiB max (not copied)\n",
            (unsigned long long)g_model_wrap_count,
            ds4_gpu_gib(g_model_wrap_bytes),
            ds4_gpu_gib(g_model_wrap_max_bytes));
    fprintf(stderr,
            "ds4:   model residency requests %llu%s\n",
            (unsigned long long)g_model_residency_count,
            getenv("DS4_METAL_NO_RESIDENCY") != NULL ? " (disabled)" : "");
    fprintf(stderr,
            "ds4:   device %s, Metal 4 runtime %s, family %s, MTL4 queue %s, tensor API %s, M5 neural accelerators %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_runtime_available ? "yes" : "no",
            g_metal4_family_supported ? "yes" : "no",
            g_metal4_queue_supported ? "yes" : "no",
            g_metal4_tensor_api_enabled ? "enabled" :
                (g_metal4_tensor_api_compile_supported ? "available" : "disabled"),
            g_metal4_m5_neural_accelerators_hint ? "likely" : "not detected");
    fprintf(stderr,
            "ds4:   accelerated Metal path %s%s\n",
            ds4_gpu_mpp_available() ? "enabled" : "disabled",
            g_quality_mode ? " by --quality" :
                (!g_metal4_tensor_api_enabled ? " (tensor API unavailable)" : ""));
    fprintf(stderr,
            "ds4:   scratch %.2f MiB (flash mask %.2f, pad %.2f, tmp %.2f, blk %.2f, ring %.2f, kv %.2f, compressor %.2f, router %.2f, indexer %.2f, moe %.2f, f16 %.2f, raw-store %.2f)\n",
            ds4_gpu_mib(scratch),
            ds4_gpu_mib((uint64_t)g_flash_attn_mask_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_pad_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_tmp_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_blk_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_ring_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_kv_bytes),
            ds4_gpu_mib((uint64_t)g_compressor_pool_kv_bytes +
                          (uint64_t)g_compressor_pool_score_bytes +
                          (uint64_t)g_compressor_pool_score_cont_bytes +
                          (uint64_t)g_compressor_pool_softmax_bytes +
                          (uint64_t)g_compressor_pool_product_bytes +
                          (uint64_t)g_compressor_store_ape_bytes +
                          (uint64_t)g_compressor_store_score_bytes +
                          (uint64_t)g_embed_rows_bytes),
            ds4_gpu_mib((uint64_t)g_router_selection_bytes +
                          (uint64_t)g_router_weight_sum_bytes),
            ds4_gpu_mib((uint64_t)g_indexer_head_scores_bytes +
                          (uint64_t)g_indexer_topk_bytes +
                          (uint64_t)g_indexed_topk_bytes),
            ds4_gpu_mib((uint64_t)g_moe_gate_scratch_bytes +
                          (uint64_t)g_moe_down_scratch_bytes +
                          (uint64_t)g_moe_id_map_bytes),
            ds4_gpu_mib((uint64_t)g_f16_round_scratch_bytes),
            ds4_gpu_mib((uint64_t)g_raw_store_round_bytes));
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static const char *ds4_gpu_source =
"#include <metal_stdlib>\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"\n";

static NSString *ds4_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:ds4_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  Environment overrides are still honored so a diagnostic
     * run can swap one source file without changing the executable.
     */
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"DS4_METAL_FLASH_ATTN_SOURCE", @"metal/flash_attn.metal"],
        @[@"DS4_METAL_DENSE_SOURCE",      @"metal/dense.metal"],
        @[@"DS4_METAL_MOE_SOURCE",        @"metal/moe.metal"],
        @[@"DS4_METAL_DSV4_HC_SOURCE",    @"metal/dsv4_hc.metal"],
        @[@"DS4_METAL_UNARY_SOURCE",      @"metal/unary.metal"],
        @[@"DS4_METAL_DSV4_KV_SOURCE",    @"metal/dsv4_kv.metal"],
        @[@"DS4_METAL_DSV4_ROPE_SOURCE",  @"metal/dsv4_rope.metal"],
        @[@"DS4_METAL_DSV4_MISC_SOURCE",  @"metal/dsv4_misc.metal"],
        @[@"DS4_METAL_ARGSORT_SOURCE",    @"metal/argsort.metal"],
        @[@"DS4_METAL_CPY_SOURCE",        @"metal/cpy.metal"],
        @[@"DS4_METAL_CONCAT_SOURCE",     @"metal/concat.metal"],
        @[@"DS4_METAL_GET_ROWS_SOURCE",   @"metal/get_rows.metal"],
        @[@"DS4_METAL_SUM_ROWS_SOURCE",   @"metal/sum_rows.metal"],
        @[@"DS4_METAL_SOFTMAX_SOURCE",    @"metal/softmax.metal"],
        @[@"DS4_METAL_REPEAT_SOURCE",     @"metal/repeat.metal"],
        @[@"DS4_METAL_GLU_SOURCE",        @"metal/glu.metal"],
        @[@"DS4_METAL_NORM_SOURCE",       @"metal/norm.metal"],
        @[@"DS4_METAL_BIN_SOURCE",        @"metal/bin.metal"],
        @[@"DS4_METAL_SET_ROWS_SOURCE",   @"metal/set_rows.metal"],
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv([spec[0] UTF8String]);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        if (override_path && override_path[0]) {
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        }
        [paths addObject:spec[1]];
        [paths addObject:[@"./" stringByAppendingString:spec[1]]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "ds4: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            fprintf(stderr,
                    "ds4: Metal source %s not found (set %s to override)\n",
                    [spec[1] UTF8String], [spec[0] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

typedef struct {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_get_rows_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_repeat_args;

typedef struct {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_set_rows_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  dim;
} ds4_gpu_concat_args;

typedef struct {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_cpy_args;

static ds4_gpu_cpy_args ds4_gpu_make_cpy_1d_args(
        uint32_t n,
        uint64_t src_elem,
        uint64_t dst_elem) {
    return (ds4_gpu_cpy_args) {
        .nk0 = (int64_t)n,
        .ne00 = (int64_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = src_elem,
        .nb01 = (uint64_t)n * src_elem,
        .nb02 = (uint64_t)n * src_elem,
        .nb03 = (uint64_t)n * src_elem,
        .ne0 = (int64_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = dst_elem,
        .nb1 = (uint64_t)n * dst_elem,
        .nb2 = (uint64_t)n * dst_elem,
        .nb3 = (uint64_t)n * dst_elem,
    };
}

static NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static float ds4_gpu_negative_infinity(void) {
    union { uint32_t u; float f; } v = { 0xff800000u };
    return v.f;
}

static float ds4_gpu_positive_infinity(void) {
    union { uint32_t u; float f; } v = { 0x7f800000u };
    return v.f;
}

static int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride);

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value);

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n);

typedef struct {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} ds4_gpu_glu_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} ds4_gpu_bin_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
} ds4_gpu_unary_args;

static ds4_gpu_bin_args ds4_gpu_make_bin_rows_args(uint32_t n, uint32_t rows, uint32_t rhs_n) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t rhs_row_bytes = (uint64_t)rhs_n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)rhs_n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = rhs_row_bytes,
        .nb13 = rhs_row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_unary_args ds4_gpu_make_unary_rows_args(
        uint32_t n,
        uint32_t rows,
        int      c4,
        float    scale,
        float    bias) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint32_t n_kernel = c4 ? n / 4u : n;
    return (ds4_gpu_unary_args) {
        .ne00 = (int32_t)n_kernel,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne0 = (int32_t)n_kernel,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .slope = 0.0f,
        .scale = scale,
        .bias = bias,
        .val = 0.0f,
        .min = 0.0f,
        .max = 0.0f,
    };
}

static ds4_gpu_bin_args ds4_gpu_make_bin_same_rows_args(uint32_t n, uint32_t rows) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = (uint64_t)rows * row_bytes,
        .nb03 = (uint64_t)rows * row_bytes,
        .ne10 = (int32_t)n,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = (uint64_t)rows * row_bytes,
        .nb13 = (uint64_t)rows * row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = (uint64_t)rows * row_bytes,
        .nb3 = (uint64_t)rows * row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off);

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows);

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_q8_0_matvec_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mm_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mv_ext_args;

typedef ds4_gpu_q8_0_matvec_args ds4_gpu_f16_matvec_args;

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_f16_matvec_args ds4_gpu_make_f16_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    return (ds4_gpu_f16_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_f32_mv_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_vec) {
    const uint64_t row_bytes = in_dim * sizeof(float);
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_vec,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_vec * sizeof(float),
        .nb13 = in_dim * n_vec * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_vec,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

typedef struct {
    const char *function_name;
    int16_t     nsg;
    int32_t     nr0;
    NSUInteger  smem;
} ds4_gpu_mv_dispatch;

static ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_mv_dispatch(void) {
    return (ds4_gpu_mv_dispatch) {
        .function_name = "kernel_mul_mv_q8_0_f32",
        .nsg = 4,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mv_dispatch ds4_gpu_make_plain_mv_dispatch(
        uint64_t in_dim,
        int      f32_weights) {
    if (in_dim < 32) {
        return (ds4_gpu_mv_dispatch) {
            .function_name = f32_weights ? "kernel_mul_mv_f32_f32_short" : "kernel_mul_mv_f16_f32_short",
            .nsg = 1,
            .nr0 = 32,
            .smem = 0,
        };
    }

    const int16_t nsg = (int16_t)((in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u);
    const int use_4 = (in_dim % 4u) == 0;
    return (ds4_gpu_mv_dispatch) {
        .function_name = f32_weights
            ? (use_4 ? "kernel_mul_mv_f32_f32_4" : "kernel_mul_mv_f32_f32")
            : (use_4 ? "kernel_mul_mv_f16_f32_4" : "kernel_mul_mv_f16_f32"),
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mul_mm_args ds4_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

static int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return 0;
    }
}

static const char *ds4_gpu_mv_ext_name(int q8, int16_t r1ptg) {
    if (q8) {
        switch (r1ptg) {
        case 2: return "kernel_mul_mv_ext_q8_0_f32_r1_2";
        case 3: return "kernel_mul_mv_ext_q8_0_f32_r1_3";
        case 4: return "kernel_mul_mv_ext_q8_0_f32_r1_4";
        case 5: return "kernel_mul_mv_ext_q8_0_f32_r1_5";
        default: return NULL;
        }
    }

    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f16_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f16_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f16_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f16_f32_r1_5";
    default: return NULL;
    }
}

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ds4_gpu_rms_norm_args;

typedef struct {
    int32_t  q_n;
    int32_t  q_n4;
    int32_t  kv_n;
    int32_t  kv_n4;
    uint64_t q_row_stride;
    uint64_t kv_row_stride;
    float    eps;
} ds4_gpu_qkv_rms_norm_args;

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_3d_args(
        uint32_t n0,
        uint32_t n1,
        uint32_t n2,
        float    eps) {
    const uint64_t row_bytes = (uint64_t)n0 * sizeof(float);
    const uint64_t plane_bytes = row_bytes * n1;
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n0,
        .ne00_t = (int32_t)(n0 / 4u),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = plane_bytes * n2,
        .eps = eps,
        .nef1 = { (int32_t)n1, 1, 1 },
        .nef2 = { (int32_t)n2, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { plane_bytes, row_bytes, row_bytes },
        .nbf3 = { plane_bytes * n2, row_bytes, row_bytes },
    };
}

static NSUInteger ds4_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

static NSUInteger ds4_gpu_rms_norm_pipeline_threads(
        uint32_t                  n,
        id<MTLComputePipelineState> pipeline) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger max_threads = pipeline ? [pipeline maxTotalThreadsPerThreadgroup] : 1024u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

typedef struct {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
} ds4_gpu_hc_split_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_hc_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float eps;
} ds4_gpu_hc_split_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float eps;
    float norm_eps;
} ds4_gpu_hc_split_weighted_sum_norm_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t has_add;
} ds4_gpu_hc_expand_args;

typedef struct {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
} ds4_gpu_mul_mv_id_args;

typedef struct {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
} ds4_gpu_mul_mm_id_map_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mm_id_args;

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0);

static int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg);

static int ds4_gpu_encode_attn_out_low_q8_mpp(
        id<MTLCommandBuffer>           cb,
        id<MTLComputePipelineState>    pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>                  src0,
        NSUInteger                     src0_off,
        id<MTLBuffer>                  src1,
        NSUInteger                     src1_off,
        id<MTLBuffer>                  dst,
        NSUInteger                     dst_off);

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);
static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size);

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off);
static int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off);

typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_pad_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_blk_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ds4_gpu_flash_attn_vec_args;

typedef struct {
    int32_t nrows;
} ds4_gpu_flash_attn_reduce_args;

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    bool src2;
} ds4_gpu_rope_tail_batch_args;

static ds4_gpu_rope_tail_batch_args ds4_gpu_make_rope_tail_args(
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        bool     inverse,
        float    freq_base,
        float    freq_scale,
        float    ext_factor,
        float    attn_factor,
        float    beta_fast,
        float    beta_slow) {
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t tok_bytes = (uint64_t)n_head * row_bytes;
    return (ds4_gpu_rope_tail_batch_args) {
        .ne00 = head_dim,
        .ne01 = n_head,
        .ne02 = n_tok,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = tok_bytes,
        .nb03 = (uint64_t)n_tok * tok_bytes,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = tok_bytes,
        .nb3 = (uint64_t)n_tok * tok_bytes,
        .n_dims = (int32_t)n_rot,
        .mode = 0,
        .n_ctx_orig = (int32_t)n_ctx_orig,
        .inverse = inverse ? 1 : 0,
        .freq_base = freq_base,
        .freq_scale = freq_scale,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .src2 = false,
    };
}

static int ds4_gpu_encode_rope_tail_inplace(
        id<MTLCommandBuffer>                 cb,
        id<MTLBuffer>                        xbuf,
        NSUInteger                           xoff,
        const ds4_gpu_rope_tail_batch_args *args,
        uint32_t                             n_tok,
        uint32_t                             n_head,
        uint32_t                             head_dim,
        uint32_t                             pos0,
        uint32_t                             pos_step) {
    int32_t pos_stack[256];
    int32_t *pos = pos_stack;
    if (n_tok > (uint32_t)(sizeof(pos_stack) / sizeof(pos_stack[0]))) {
        pos = malloc((size_t)n_tok * sizeof(*pos));
        if (!pos) {
            fprintf(stderr, "ds4: failed to allocate Metal RoPE position buffer\n");
            return 0;
        }
    }
    for (uint32_t t = 0; t < n_tok; t++) pos[t] = (int32_t)(pos0 + t * pos_step);

    const NSUInteger pos_bytes = (NSUInteger)n_tok * sizeof(*pos);
    id<MTLBuffer> posbuf = nil;
    if (pos_bytes > 4096u) {
        /*
         * Metal inline setBytes data is meant for small constants. Long prefill
         * RoPE calls need thousands of positions; passing that much inline can
         * make the Apple driver abort the process instead of reporting a normal
         * API error.
         */
        posbuf = ds4_gpu_new_transient_buffer(pos_bytes, "ds4_rope_positions");
        if (!posbuf) {
            if (pos != pos_stack) free(pos);
            return 0;
        }
        memcpy([posbuf contents], pos, pos_bytes);
    }

    const NSUInteger nth = (NSUInteger)(head_dim < 256u ? head_dim : 256u);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_rope_tail_batch_pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    if (posbuf) {
        [enc setBuffer:posbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:pos length:pos_bytes atIndex:2];
    }
    [enc setBuffer:xbuf offset:xoff atIndex:3];
    [enc setBuffer:xbuf offset:xoff atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
         threadsPerThreadgroup:MTLSizeMake(nth ? nth : 1u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    if (pos != pos_stack) free(pos);
    return 1;
}

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_rot;
} ds4_gpu_dsv4_fp8_kv_quantize_args;

typedef struct {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
} ds4_gpu_dsv4_kv_fp8_store_args;

typedef struct {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
} ds4_gpu_dsv4_indexer_qat_args;

typedef struct {
    uint32_t width;
} ds4_gpu_dsv4_ratio4_shift_args;

typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
} ds4_gpu_dsv4_compressor_store_one_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_softmax_pool_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
} ds4_gpu_kargs_argsort;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
} ds4_gpu_kargs_argsort_merge;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_kargs_sum_rows;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
} ds4_gpu_softmax_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_topk_mask_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
} ds4_gpu_dsv4_indexer_weighted_sum_args;

typedef struct {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
} ds4_gpu_dsv4_router_select_one_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
} ds4_gpu_dsv4_indexed_attention_args;

typedef struct {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
} ds4_gpu_dsv4_indexer_scores_fused_args;

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float    clamp_value;
} ds4_gpu_dsv4_moe_swiglu_weight_args;

typedef struct {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
} ds4_gpu_dsv4_moe_sum6_args;

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small ds4_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants. */
int ds4_gpu_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "ds4: Metal device not available\n");
            return 0;
        }
        ds4_gpu_print_device_summary();
        ds4_gpu_detect_metal4_features();

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "ds4: failed to create Metal command queue\n");
            g_device = nil;
            return 0;
        }
        g_model_buffer_cache = [NSMutableDictionary dictionary];
        g_pipeline_cache = [NSMutableDictionary dictionary];
        g_transient_buffers = [NSMutableArray array];
        g_pending_cbs = [NSMutableArray array];
        if (!g_model_buffer_cache || !g_pipeline_cache || !g_transient_buffers || !g_pending_cbs) {
            fprintf(stderr, "ds4: Metal bookkeeping allocation failed\n");
            g_pending_cbs = nil;
            g_transient_buffers = nil;
            g_pipeline_cache = nil;
            g_model_buffer_cache = nil;
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        NSError *error = nil;
        NSString *source = ds4_gpu_full_source();
        if (!source) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        NSMutableDictionary *macros = [NSMutableDictionary new];
        if (g_metal4_tensor_api_enabled) {
            macros[@"DS4_METAL_HAS_TENSOR"] = @"1";
            fprintf(stderr, "ds4: Metal 4 tensor API enabled for Tensor kernels\n");
        }

        const int drift_hc_stable        = ds4_gpu_env_bool("DS4_METAL_HC_STABLE")          != 0; // default ON
        const int drift_norm_unify       = ds4_gpu_env_bool("DS4_METAL_NORM_RSQRT_DISABLE") != 0; // default ON
        const int drift_kv_raw_f32       = ds4_gpu_env_bool("DS4_METAL_KV_RAW_F32")         >  0; // default OFF
        const int drift_rope_exp2_log2   = ds4_gpu_env_bool("DS4_METAL_ROPE_EXP2_LOG2")     >  0; // default OFF
        const int drift_math_safe        = ds4_gpu_env_bool("DS4_METAL_MATH_SAFE")          >  0; // default OFF

        if (drift_math_safe) {
            // MTLCompileOptions.fastMathEnabled defaults to YES and Apple's
            // headers explicitly say this "may violate the IEEE 754 standard".
            // Different fast-math optimizations get applied across the
            // matmul2d cooperative-tensor path and the legacy
            // simdgroup_multiply_accumulate path on M5, amplifying the
            // mismatch. MTLMathModeSafe pins the entire library to strict
            // IEEE-754 semantics. Diagnostic-only: useful to localize drift
            // sources but not to ship as a default.
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                fprintf(stderr, "ds4: Metal shader library math mode = safe (strict IEEE-754) by DS4_METAL_MATH_SAFE\n");
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                fprintf(stderr, "ds4: Metal shader library fast-math disabled by DS4_METAL_MATH_SAFE (pre-macOS 15)\n");
            }
        }

        if (drift_hc_stable)      macros[@"DS4_METAL_HC_STABLE"]          = @"1";
        if (drift_norm_unify)     macros[@"DS4_METAL_NORM_RSQRT_DISABLE"] = @"1";
        if (drift_kv_raw_f32)     macros[@"DS4_METAL_KV_RAW_F32"]         = @"1";
        if (drift_rope_exp2_log2) macros[@"DS4_METAL_ROPE_EXP2_LOG2"]     = @"1";
        fprintf(stderr,
                "ds4: drift-patch flags hc_stable=%s norm_unify=%s kv_raw_f32=%s rope_exp2_log2=%s math_safe=%s tensor_matmul=%s\n",
                drift_hc_stable      ? "on"  : "off",
                drift_norm_unify     ? "on"  : "off",
                drift_kv_raw_f32     ? "on"  : "off",
                drift_rope_exp2_log2 ? "on"  : "off",
                drift_math_safe      ? "on"  : "off",
                g_metal4_tensor_api_enabled ? "on" : "off");
        options.preprocessorMacros = macros;
        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "ds4: Metal shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_library = library;

        id<MTLFunction> fn = [library newFunctionWithName:@"kernel_get_rows_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_repeat_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_repeat_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_repeat_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_set_rows_f32_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_set_rows_f32_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_set_rows_f32_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_concat"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_concat function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_concat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_concat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_concat pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f16_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f16_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_fp8_kv_quantize_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_fp8_kv_quantize_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_fp8_kv_quantize_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_hadamard_fp4_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_qat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_qat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_kv_fp8_store_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_kv_fp8_store_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_kv_fp8_store_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_ratio4_shift_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_ratio4_shift_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_ratio4_shift_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_moe_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_op = 0;
        int16_t bin_f = 1;
        bool bin_rb = false;
        bool bin_cb = false;
        [bin_constants setConstantValue:&bin_op type:MTLDataTypeShort atIndex:1300];
        [bin_constants setConstantValue:&bin_f  type:MTLDataTypeShort atIndex:1301];
        [bin_constants setConstantValue:&bin_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_constants setConstantValue:&bin_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_add_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_plain_op = 2;
        int16_t bin_mul_plain_f = 1;
        bool bin_mul_plain_rb = false;
        bool bin_mul_plain_cb = false;
        [bin_mul_constants setConstantValue:&bin_mul_plain_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_constants setConstantValue:&bin_mul_plain_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_constants setConstantValue:&bin_mul_plain_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_constants setConstantValue:&bin_mul_plain_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_mul_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_scalar_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_op = 2;
        int16_t bin_mul_f = 1;
        bool bin_mul_rb = false;
        bool bin_mul_cb = true;
        [bin_mul_scalar_constants setConstantValue:&bin_mul_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_scalar_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_mul_scalar_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_mul_scalar_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_div_row_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_div_op = 3;
        int16_t bin_div_f = 1;
        bool bin_div_rb = false;
        bool bin_div_cb = true;
        [bin_div_row_constants setConstantValue:&bin_div_op type:MTLDataTypeShort atIndex:1300];
        [bin_div_row_constants setConstantValue:&bin_div_f  type:MTLDataTypeShort atIndex:1301];
        [bin_div_row_constants setConstantValue:&bin_div_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_div_row_constants setConstantValue:&bin_div_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_div_row_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_div_row_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_div_row_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_plain_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_plain_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_qkv_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_qkv_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_qkv_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t moe_mv_id_nsg = 2;
        [moe_mv_id_constants setConstantValue:&moe_mv_id_nsg type:MTLDataTypeShort atIndex:600];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rope_tail_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_batch_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_softmax_pool"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_softmax_pool_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_softmax_pool_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_4_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_4_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_merge_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_merge_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_merge_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *sum_rows_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t sum_rows_op = 10;
        [sum_rows_constants setConstantValue:&sum_rows_op type:MTLDataTypeShort atIndex:1400];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                           constantValues:sum_rows_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_sum_rows_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_sum_rows_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask_scatter"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_scatter_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_scatter_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_sinkhorn"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_sinkhorn_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_sinkhorn_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum_norm4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sigmoid_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sigmoid_op = 102;
        bool unary_cnt = false;
        [unary_sigmoid_constants setConstantValue:&unary_sigmoid_op type:MTLDataTypeShort atIndex:1200];
        [unary_sigmoid_constants setConstantValue:&unary_cnt        type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sigmoid_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sigmoid_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sigmoid_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_silu_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_silu_op = 106;
        [unary_silu_constants setConstantValue:&unary_silu_op type:MTLDataTypeShort atIndex:1200];
        [unary_silu_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_silu_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_silu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_softplus_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_softplus_op = 115;
        [unary_softplus_constants setConstantValue:&unary_softplus_op type:MTLDataTypeShort atIndex:1200];
        [unary_softplus_constants setConstantValue:&unary_cnt         type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_softplus_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_softplus_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_softplus_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sqrt_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sqrt_op = 14;
        [unary_sqrt_constants setConstantValue:&unary_sqrt_op type:MTLDataTypeShort atIndex:1200];
        [unary_sqrt_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sqrt_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sqrt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sqrt_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_clamp_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_clamp_op = 12;
        [unary_clamp_constants setConstantValue:&unary_clamp_op type:MTLDataTypeShort atIndex:1200];
        [unary_clamp_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32"
                           constantValues:unary_clamp_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_clamp_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_clamp_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_scale_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_scale_op = 10;
        [unary_scale_constants setConstantValue:&unary_scale_op type:MTLDataTypeShort atIndex:1200];
        [unary_scale_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_scale_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_scale_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_fill_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_fill_op = 11;
        [unary_fill_constants setConstantValue:&unary_fill_op type:MTLDataTypeShort atIndex:1200];
        [unary_fill_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f16_f16"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_expand"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_expand_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_expand_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_dsv4_indexer_score_one_direct_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexer_score_one_direct");
        g_dsv4_compressor_store_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_compressor_store_one");
        g_dsv4_sort_i32_rows_asc_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_sort_i32_rows_asc");
        g_dsv4_indexed_attention_heads8_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8");
        g_dsv4_indexed_attention_heads8_rb16_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb16");
        g_dsv4_softplus_sqrt_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_softplus_sqrt_f32_4");
        g_dsv4_router_finalize_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one");
        g_dsv4_router_weights_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_weights_one");
        g_dsv4_route_translate_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_route_translate");
        g_dsv4_hc_expand4_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_hc_expand4");
        if (!g_dsv4_indexer_score_one_direct_pipeline ||
            !g_dsv4_compressor_store_one_pipeline ||
            !g_dsv4_sort_i32_rows_asc_pipeline ||
            !g_dsv4_indexed_attention_heads8_pipeline ||
            !g_dsv4_indexed_attention_heads8_rb16_pipeline ||
            !g_dsv4_softplus_sqrt_pipeline ||
            !g_dsv4_router_finalize_one_pipeline ||
            !g_dsv4_router_weights_one_pipeline ||
            !g_dsv4_route_translate_pipeline ||
            !g_dsv4_hc_expand4_pipeline) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
    }

    return 1;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *tensor = [DS4MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) {
            return NULL;
        }
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        g_tensor_alloc_live_bytes += bytes;
        if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
            g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
        }
        if (ds4_gpu_trace_allocs()) {
            fprintf(stderr,
                    "ds4: Metal tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    (double)bytes / (1024.0 * 1024.0),
                    (double)g_tensor_alloc_live_bytes / (1024.0 * 1024.0),
                    (double)g_tensor_alloc_peak_bytes / (1024.0 * 1024.0));
        }
        return (__bridge_retained ds4_gpu_tensor *)tensor;
    }
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    (void)kv_cache_bytes;
    (void)context_bytes;
    return 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    const DS4MetalTensor *base_obj = ds4_gpu_tensor_const_obj(base);
    if (offset > base_obj.bytes || bytes > base_obj.bytes - offset) return NULL;
    if (base_obj.offset > UINT64_MAX - offset) return NULL;
    const uint64_t absolute_offset = base_obj.offset + offset;
    if (absolute_offset > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = base_obj.buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        return (__bridge_retained ds4_gpu_tensor *)view;
    }
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        DS4MetalTensor *obj = (__bridge_transfer DS4MetalTensor *)tensor;
        if (obj.owner) {
            if (obj.bytes <= g_tensor_alloc_live_bytes) {
                g_tensor_alloc_live_bytes -= obj.bytes;
            } else {
                g_tensor_alloc_live_bytes = 0;
            }
            if (ds4_gpu_trace_allocs()) {
                fprintf(stderr,
                        "ds4: Metal tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                        (double)obj.bytes / (1024.0 * 1024.0),
                        (double)g_tensor_alloc_live_bytes / (1024.0 * 1024.0),
                        (double)g_tensor_alloc_peak_bytes / (1024.0 * 1024.0));
            }
        }
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.bytes;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    return (uint8_t *)[obj.buffer contents] + obj.offset;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > ds4_gpu_tensor_bytes(tensor) / sizeof(float)) return 0;
    float *p = ds4_gpu_tensor_contents(tensor);
    if (!p && count != 0) return 0;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)[obj.buffer contents] + obj.offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)[obj.buffer contents] + obj.offset + offset, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (dst_offset > d.bytes || bytes > d.bytes - dst_offset) return 0;
    if (src_offset > s.bytes || bytes > s.bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:s.buffer
            sourceOffset:(NSUInteger)(s.offset + src_offset)
                toBuffer:d.buffer
       destinationOffset:(NSUInteger)(d.offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(uint16_t);
    if (src_offset > s.bytes || src_bytes > s.bytes - src_offset ||
        dst_offset > d.bytes || dst_bytes > d.bytes - dst_offset) {
        return 0;
    }

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        uint64_t done = 0;
        int ok = 1;
        while (done < count && ok) {
            uint64_t chunk64 = count - done;
            if (chunk64 > UINT32_MAX) chunk64 = UINT32_MAX;
            const uint32_t chunk = (uint32_t)chunk64;
            ok = ds4_gpu_encode_cpy_f32_f16_1d(
                    cb,
                    s.buffer,
                    (NSUInteger)(s.offset + src_offset + done * sizeof(float)),
                    d.buffer,
                    (NSUInteger)(d.offset + dst_offset + done * sizeof(uint16_t)),
                    chunk);
            done += chunk;
        }
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "tensor f32 to f16 copy");
        return ok;
    }
}

int ds4_gpu_begin_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_cb = [g_queue commandBuffer];
    return g_batch_cb != nil;
}

int ds4_gpu_flush_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    [g_pending_cbs addObject:cb];

    g_batch_cb = [g_queue commandBuffer];
    if (!g_batch_cb) {
        (void)ds4_gpu_wait_pending_command_buffers("command batch");
        [g_transient_buffers removeAllObjects];
        return 0;
    }
    return 1;
}

int ds4_gpu_end_commands(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    return ds4_gpu_finish_command_buffer(cb, 1, "command batch");
}

/* project.md P0.1 barrier-tax recovery: the A3 expert path drains the batch
 * before every routed layer's CPU gather (the gather must read this layer's
 * router output, and may overwrite scratch the previous MoE still reads).
 * The classic drain commits + waitUntilCompleted, paying the slow per-CB
 * scheduling/status path every routed layer (~43x per decoded token).  The
 * fast drain reuses the MTLSharedEvent host-wait fast path (same precedent as
 * the TP rendezvous below: ~150ms -> <50us per sync): signal at the end of
 * the batch, commit without waiting, and block on waitUntilSignaledValue,
 * which gives the same "all prior GPU work complete, writes visible"
 * guarantee.  Command-buffer status/error checking is deferred one drain: the
 * CB is pushed on g_pending_cbs and swept by the next drain after its event
 * wait, when the CB is already complete and the sweep is free. */
static id<MTLSharedEvent> g_a3_drain_event;
static uint64_t g_a3_drain_event_value;

static int ds4_gpu_expert_event_drain_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_EVENT_DRAIN") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: A3 expert drain uses MTLSharedEvent fast host wait "
                    "(DS4_METAL_EXPERT_EVENT_DRAIN=1)\n");
        }
    }
    return cached;
}

static int ds4_gpu_end_commands_event(const char *label) {
    if (!g_batch_cb) return 0;
    if (!g_a3_drain_event) {
        g_a3_drain_event = [g_device newSharedEvent];
        if (!g_a3_drain_event) return ds4_gpu_end_commands();
    }
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    const uint64_t value = ++g_a3_drain_event_value;
    [cb encodeSignalEvent:g_a3_drain_event value:value];
    [cb commit];
    if (![g_a3_drain_event waitUntilSignaledValue:value timeoutMS:60000]) {
        fprintf(stderr,
                "ds4: Metal %s event drain timed out; falling back to waitUntilCompleted\n",
                label);
        if (!ds4_gpu_wait_command_buffer(cb, label)) return 0;
    }
    /* Everything queued before the signal is complete: sweep the deferred
     * status checks (instant now) and release transient buffer references. */
    int ok = ds4_gpu_wait_pending_command_buffers(label);
    [g_pending_cbs addObject:cb];
    [g_transient_buffers removeAllObjects];
    return ok;
}

static int ds4_gpu_expert_drain_commands(const char *label) {
    if (ds4_gpu_expert_event_drain_enabled()) return ds4_gpu_end_commands_event(label);
    return ds4_gpu_end_commands();
}

/* Tensor-parallel host/GPU rendezvous. Instead of draining the whole pipeline
 * with waitUntilCompleted (slow per-CB scheduling path), we signal a
 * MTLSharedEvent at the end of the current batch and let the host wait on that
 * specific value — the MTLSharedEvent.waitUntilSignaledValue fast path
 * (Anukari/Apple precedent: ~150ms -> <50us per sync). The caller flushes the
 * batch (commit, no wait) right after signalling so the GPU runs and fires the
 * event while the host proceeds. */
static id<MTLSharedEvent> g_tp_event;
static uint64_t g_tp_event_value;

uint64_t ds4_gpu_tp_signal_after_batch(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    if (!g_tp_event) {
        g_tp_event = [g_device newSharedEvent];
        if (!g_tp_event) return 0;
    }
    uint64_t value = ++g_tp_event_value; /* values are reserved nonzero (0 == error) */
    [g_batch_cb encodeSignalEvent:g_tp_event value:value];
    return value;
}

int ds4_gpu_tp_host_wait(uint64_t value) {
    if (!g_tp_event || value == 0) return 0;
    uint64_t timeout_ms = 60000;
    const char *env = getenv("DS4_TP_EVENT_TIMEOUT_MS");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && v > 0) timeout_ms = (uint64_t)v;
    }
    return [g_tp_event waitUntilSignaledValue:value timeoutMS:timeout_ms] ? 1 : 0;
}

static int ds4_gpu_flash_attn_stage_profile_boundary(
        id<MTLCommandBuffer> __strong *cbp,
        const char           *mode,
        const char           *stage,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              n_keys,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              window,
        uint32_t              ratio,
        double               *stage_t0) {
    if (!cbp || !*cbp || !stage_t0 || !stage) return 0;
    if (ds4_gpu_end_commands() == 0) return 0;

    const double now_ms = ds4_gpu_now_ms();
    const char *filter = getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER");
    const int print_stage =
        !filter || !filter[0] ||
        strstr(stage, filter) != NULL ||
        (mode && strstr(mode, filter) != NULL);
    if (print_stage) {
        fprintf(stderr,
                "ds4: Metal FlashAttention prefill stage mode=%s tokens=%u comp=%u "
                "keys=%u heads=%u dim=%u window=%u ratio=%u %s=%.3f ms\n",
                mode ? mode : "unknown",
                n_tokens,
                n_comp,
                n_keys,
                n_head,
                head_dim,
                window,
                ratio,
                stage,
                now_ms - *stage_t0);
    }
    *stage_t0 = now_ms;

    if (ds4_gpu_begin_commands() == 0) return 0;
    int owned = 0;
    *cbp = ds4_gpu_command_buffer(&owned);
    return *cbp != nil && owned == 0;
}

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return ds4_gpu_end_commands();
    if ([g_pending_cbs count] != 0) {
        int ok = ds4_gpu_wait_pending_command_buffers("synchronize");
        [g_transient_buffers removeAllObjects];
        return ok;
    }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;
    return ds4_gpu_finish_command_buffer(cb, 1, "synchronize");
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        if (g_batch_cb) {
            ds4_gpu_close_batch_encoder();
            [g_batch_cb commit];
            [g_batch_cb waitUntilCompleted];
            g_batch_cb = nil;
        }
        (void)ds4_gpu_wait_pending_command_buffers("cleanup");
        [g_transient_buffers removeAllObjects];
        g_tp_event = nil;
        g_tp_event_value = 0;
        g_a3_drain_event = nil;
        g_a3_drain_event_value = 0;
        g_set_rows_f32_i32_pipeline = nil;
        g_get_rows_f32_pipeline = nil;
        g_get_rows_f16_pipeline = nil;
        g_get_rows_i32_pipeline = nil;
        g_repeat_f32_pipeline = nil;
        g_concat_pipeline = nil;
        g_cpy_f32_f32_pipeline = nil;
        g_cpy_f32_f16_pipeline = nil;
        g_cpy_f16_f32_pipeline = nil;
        g_swiglu_pipeline = nil;
        g_add_pipeline = nil;
        g_moe_sum6_pipeline = nil;
        g_mul_pipeline = nil;
        g_bin_mul_scalar_pipeline = nil;
        g_bin_div_row_pipeline = nil;
        g_unary_sigmoid_pipeline = nil;
        g_unary_silu_pipeline = nil;
        g_unary_softplus_pipeline = nil;
        g_unary_sqrt_pipeline = nil;
        g_unary_clamp_pipeline = nil;
        g_unary_scale_pipeline = nil;
        g_unary_fill_pipeline = nil;
        g_unary_fill_f16_pipeline = nil;
        g_rms_norm_pipeline = nil;
        g_rms_norm_plain_pipeline = nil;
        g_dsv4_qkv_rms_norm_pipeline = nil;
        g_hc_split_sinkhorn_pipeline = nil;
        g_hc_split_weighted_sum_pipeline = nil;
        g_hc_split_weighted_sum_norm_pipeline = nil;
        g_hc_weighted_sum_pipeline = nil;
        g_hc_expand_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q2_k_pipeline = nil;
        g_moe_mul_mv_id_q2_k_sum6_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q4_k_sum6_pipeline = nil;
        g_rope_tail_batch_pipeline = nil;
        g_dsv4_fp8_kv_quantize_pipeline = nil;
        g_dsv4_indexer_qat_pipeline = nil;
        g_dsv4_kv_fp8_store_pipeline = nil;
        g_dsv4_ratio4_shift_pipeline = nil;
        g_dsv4_softmax_pool_pipeline = nil;
        g_soft_max_f32_pipeline = nil;
        g_soft_max_f32_4_pipeline = nil;
        g_argsort_f32_i32_desc_pipeline = nil;
        g_argsort_merge_f32_i32_desc_pipeline = nil;
        g_sum_rows_f32_f32_pipeline = nil;
        g_dsv4_topk_mask_pipeline = nil;
        g_dsv4_topk_mask_scatter_pipeline = nil;
        g_dsv4_indexer_weighted_sum_pipeline = nil;
        g_dsv4_indexer_score_one_direct_pipeline = nil;
        g_dsv4_compressor_store_one_pipeline = nil;
        g_dsv4_sort_i32_rows_asc_pipeline = nil;
        g_dsv4_indexed_attention_heads8_pipeline = nil;
        g_dsv4_indexed_attention_heads8_rb16_pipeline = nil;
        g_dsv4_softplus_sqrt_pipeline = nil;
        g_dsv4_router_finalize_one_pipeline = nil;
        g_dsv4_router_weights_one_pipeline = nil;
        g_dsv4_route_translate_pipeline = nil;
        g_dsv4_hc_expand4_pipeline = nil;
        g_flash_attn_mask_buffer = nil;
        g_flash_attn_pad_buffer = nil;
        g_flash_attn_tmp_buffer = nil;
        g_flash_attn_blk_buffer = nil;
        g_flash_attn_ring_buffer = nil;
        g_flash_attn_kv_buffer = nil;
        g_compressor_pool_kv_buffer = nil;
        g_compressor_pool_score_buffer = nil;
        g_compressor_pool_score_cont_buffer = nil;
        g_compressor_pool_softmax_buffer = nil;
        g_compressor_pool_product_buffer = nil;
        g_compressor_store_ape_buffer = nil;
        g_compressor_store_score_buffer = nil;
        g_embed_rows_buffer = nil;
        g_router_selection_buffer = nil;
        g_router_weight_sum_buffer = nil;
        g_expert_keep_lut_buffer = nil;
        g_expert_keep_lut_layers = 0;
        g_indexer_head_scores_buffer = nil;
        g_indexer_topk_buffer = nil;
        g_indexed_topk_buffer = nil;
        g_f16_round_scratch_buffer = nil;
        g_raw_store_round_buffer = nil;
        g_moe_gate_scratch_buffer = nil;
        g_moe_down_scratch_buffer = nil;
        g_moe_id_map_buffer = nil;
        g_attn_out_group_ids_buffer = nil;
        g_moe_scratch_gate = nil;
        g_moe_scratch_up = nil;
        g_moe_scratch_down = nil;
        g_model_map_ptr = NULL;
        g_model_map_size = 0;
        g_model_mapped_offset = 0;
        g_model_mapped_size = 0;
        g_model_mapped_max_tensor_bytes = 0;
        g_tensor_alloc_live_bytes = 0;
        g_tensor_alloc_peak_bytes = 0;
        g_flash_attn_mask_bytes = 0;
        g_flash_attn_pad_bytes = 0;
        g_flash_attn_tmp_bytes = 0;
        g_flash_attn_blk_bytes = 0;
        g_flash_attn_ring_bytes = 0;
        g_flash_attn_kv_bytes = 0;
        g_compressor_pool_kv_bytes = 0;
        g_compressor_pool_score_bytes = 0;
        g_compressor_pool_score_cont_bytes = 0;
        g_compressor_pool_softmax_bytes = 0;
        g_compressor_pool_product_bytes = 0;
        g_compressor_store_ape_bytes = 0;
        g_compressor_store_score_bytes = 0;
        g_embed_rows_bytes = 0;
        g_router_selection_bytes = 0;
        g_router_weight_sum_bytes = 0;
        g_indexer_head_scores_bytes = 0;
        g_indexer_topk_bytes = 0;
        g_indexed_topk_bytes = 0;
        g_f16_round_scratch_bytes = 0;
        g_raw_store_round_bytes = 0;
        g_moe_gate_scratch_bytes = 0;
        g_moe_down_scratch_bytes = 0;
        g_moe_id_map_bytes = 0;
        g_attn_out_group_ids_bytes = 0;
        g_moe_scratch_gate_bytes = 0;
        g_moe_scratch_up_bytes = 0;
        g_moe_scratch_down_bytes = 0;
        g_model_wrap_count = 0;
        g_model_wrap_bytes = 0;
        g_model_wrap_max_bytes = 0;
        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();
        [g_pipeline_cache removeAllObjects];
        g_pipeline_cache = nil;
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache = nil;
        g_transient_buffers = nil;
        g_pending_cbs = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

static int ds4_gpu_encode_get_rows_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !tokens || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int32_t)n_embd,
        .ne00 = (int32_t)n_embd,
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)n_vocab * src_row_bytes,
        .nb03 = (uint64_t)n_vocab * src_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = dst_row_bytes,
        .nb2 = (uint64_t)n_tokens * dst_row_bytes,
        .nb3 = (uint64_t)n_tokens * dst_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_repeat_hc_embedding(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        rows,
        NSUInteger           rows_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_tokens,
        uint32_t             n_embd,
        uint32_t             n_hc) {
    if (!cb || !rows || !out || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;

    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    ds4_gpu_repeat_args args = {
        .ne00 = (int32_t)n_embd,
        .ne01 = 1,
        .ne02 = (int32_t)n_tokens,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = embd_bytes,
        .nb02 = embd_bytes,
        .nb03 = (uint64_t)n_tokens * embd_bytes,
        .ne0 = (int32_t)n_embd,
        .ne1 = (int32_t)n_hc,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = embd_bytes,
        .nb2 = (uint64_t)n_hc * embd_bytes,
        .nb3 = (uint64_t)n_tokens * n_hc * embd_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_repeat_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_repeat_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:rows offset:rows_offset atIndex:1];
    [enc setBuffer:out offset:out_offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(n_hc, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out_hc) < out_bytes) {
            fprintf(stderr, "ds4: Metal graph embedding received undersized HC output buffer\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger row_bytes = (NSUInteger)n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             row_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token;
        const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
        const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
        ds4_gpu_get_rows_args args = {
            .ne00t = (int32_t)n_embd,
            .ne00 = (int32_t)n_embd,
            .nb01 = src_row_bytes,
            .nb02 = (uint64_t)n_vocab * src_row_bytes,
            .nb03 = (uint64_t)n_vocab * src_row_bytes,
            .ne10 = 1,
            .nb10 = sizeof(int32_t),
            .nb11 = sizeof(int32_t),
            .nb12 = sizeof(int32_t),
            .nb1 = dst_row_bytes,
            .nb2 = dst_row_bytes,
            .nb3 = dst_row_bytes,
        };
        NSUInteger nth = (NSUInteger)n_embd;
        const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > max_threads) nth = max_threads;
        if (nth == 0) nth = 1;
        const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_get_rows_f16_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBytes:&token_i32 length:sizeof(token_i32) atIndex:2];
        [enc setBuffer:g_embed_rows_buffer offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nw0, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * n_hc * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out_hc) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal graph batched embedding received undersized buffers\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger rows_bytes = (NSUInteger)n_tokens * n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             rows_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_f16(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           g_embed_rows_buffer,
                                           0,
                                           n_vocab,
                                           n_tokens,
                                           n_embd) ||
            !ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  n_tokens,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed tokens")) return 0;
    }

    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;
    max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(map_size, max_tensor_bytes);

    @autoreleasepool {
        if (g_model_map_ptr == model_map &&
            g_model_map_size == model_size &&
            g_model_mapped_offset == map_offset &&
            g_model_mapped_size == map_size &&
            g_model_mapped_max_tensor_bytes == max_tensor_bytes) {
            return 1;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            if (g_model_views[i].model_map == model_map &&
                g_model_views[i].model_size == model_size &&
                map_offset >= g_model_views[i].model_offset &&
                map_offset + map_size <= g_model_views[i].model_offset + g_model_views[i].bytes) {
                return 1;
            }
        }

        ds4_gpu_model_residency_clear();
        if (!ds4_gpu_map_model_views(model_map, model_size, map_offset, map_size, max_tensor_bytes)) {
            ds4_gpu_model_residency_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = map_offset;
        g_model_mapped_size = map_size;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        fprintf(stderr,
                "ds4: Metal mapped mmaped model as %u overlapping shared buffers\n",
                g_model_view_count);
        return 1;
    }
}

/* Shared implementation for the plain and split span loaders. resident_flags is
 * an optional per-span bool array: when NULL every span is resident (legacy
 * behaviour); otherwise a false entry wraps that span's views without adding them
 * to the model residency set, so their clean file-backed pages stay reclaimable
 * (routed-expert offload under DS4_METAL_EXPERT_OFFLOAD). */
static int ds4_gpu_set_model_map_spans_impl(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        const bool *resident_flags,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    /* A single all-resident span is exactly the contiguous range loader; skip the
     * disjoint-buffer bookkeeping. A non-resident single span still needs the
     * per-view resident_hint, so fall through to the general path for it. */
    if (count == 1 && (!resident_flags || resident_flags[0])) {
        return ds4_gpu_set_model_map_range(model_map,
                                           model_size,
                                           offsets[0],
                                           sizes[0],
                                           max_tensor_bytes);
    }

    @autoreleasepool {
        const double t0 = ds4_gpu_now_ms();
        max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(model_size, max_tensor_bytes);

        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();

        uint64_t mapped_total = 0;
        uint64_t first_offset = UINT64_MAX;
        for (uint32_t i = 0; i < count; i++) {
            if (offsets[i] > model_size || sizes[i] == 0 || sizes[i] > model_size - offsets[i]) {
                fprintf(stderr, "ds4: Metal model span %u is outside the GGUF mapping\n", i);
                ds4_gpu_model_residency_clear();
                ds4_gpu_model_views_clear();
                return 0;
            }
            if (offsets[i] < first_offset) first_offset = offsets[i];
            uint64_t effective_max = max_tensor_bytes;
            if (effective_max > sizes[i]) effective_max = sizes[i];
            const bool span_resident = resident_flags ? resident_flags[i] : true;
            if (!ds4_gpu_add_model_view_range(model_map,
                                              model_size,
                                              offsets[i],
                                              sizes[i],
                                              effective_max,
                                              span_resident,
                                              &mapped_total)) {
                ds4_gpu_model_residency_clear();
                ds4_gpu_model_views_clear();
                return 0;
            }
        }
        if (!ds4_gpu_finish_model_views(t0, mapped_total, first_offset)) {
            ds4_gpu_model_residency_clear();
            ds4_gpu_model_views_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = first_offset == UINT64_MAX ? 0 : first_offset;
        g_model_mapped_size = mapped_total;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        fprintf(stderr,
                "ds4: Metal mapped mmaped model as %u disjoint shared buffers across %u tensor spans\n",
                g_model_view_count,
                count);
        return 1;
    }
}

int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    return ds4_gpu_set_model_map_spans_impl(model_map, model_size, offsets, sizes,
                                            NULL, count, max_tensor_bytes);
}

/* Reduced-memory loader: resident spans (backbone) are wrapped and added to the
 * residency set; non-resident spans (routed experts) are wrapped so the hot path
 * still resolves a buffer, but are kept out of the residency set so their clean
 * file-backed pages stay reclaimable. Concatenate the two span lists, mark the
 * resident ones true and the rest false, and pass one array. */
int ds4_gpu_set_model_map_spans_split(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        const bool *resident_flags,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    if (!resident_flags) return 0;   /* split loader requires explicit flags */
    return ds4_gpu_set_model_map_spans_impl(model_map, model_size, offsets, sizes,
                                            resident_flags, count, max_tensor_bytes);
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map_range(model_map, model_size, 0, model_size, 0);
}

/* Main-model GGUF file descriptor, registered by the engine right after the
 * model is opened (the whole file is mmapped from offset 0, so file offset ==
 * map offset).  The P1.1 single-copy expert gather (DS4_METAL_EXPERT_PREAD=1,
 * project.md) preads cold expert bytes straight from this fd into the Shared
 * scratch MTLBuffer, bypassing the mmap page-fault + memcpy double copy. */
static int g_model_fd = -1;
static int g_model_fd_conflict = 0;

int ds4_gpu_set_model_fd(int fd) {
    if (fd < 0) return 1;
    if (g_model_fd >= 0 && g_model_fd != fd) {
        /* Two different model files registered in one process: the pread fast
         * path can no longer tell which file backs a given map; disable it. */
        g_model_fd_conflict = 1;
    }
    g_model_fd = fd;
    return 1;
}

/* Wave 25: backbone wiring.  Verify rounds stream GiBs of one-shot expert
 * bytes and evict the mmap-resident backbone (Q8 attention/shared) pages;
 * the next frame's GPU work then re-faults the backbone inside command
 * execution (measured: coord decode drain 14.5 -> 46ms/layer, spikes 855ms).
 * Instead of bending the batch IO around the page cache, pin the backbone:
 * every model range the encoder wraps for GPU use is by definition this
 * machine's working set -- mlock it once (routed expert tensors are
 * suppressed at their wrap sites; they are gathered, not mapped).
 * DS4_METAL_BACKBONE_MLOCK=1 enables (script sets it per side);
 * DS4_METAL_BACKBONE_MLOCK_BUDGET_MB caps the wired total (default 4608). */
static int g_wrap_mlock_suppress;   /* set around routed-expert wrap calls */

static void ds4_gpu_backbone_mlock_register(const void *model_map,
                                            uint64_t offset,
                                            uint64_t len) {
    static int enabled = -1;
    static uint64_t budget_bytes;
    static uint64_t wired_bytes;
    static uint64_t logged_half_gib;
    static int budget_warned;
    static uint32_t n_ranges;
    static struct { uint64_t off, len; } ranges[4096];
    if (enabled < 0) {
        const char *v = getenv("DS4_METAL_BACKBONE_MLOCK");
        enabled = (v && *v && v[0] != '0') ? 1 : 0;
        if (enabled) {
            const char *b = getenv("DS4_METAL_BACKBONE_MLOCK_BUDGET_MB");
            uint64_t mb = b ? strtoull(b, NULL, 10) : 0;
            if (mb == 0) mb = 4608;
            budget_bytes = mb << 20;
            fprintf(stderr, "ds4: backbone mlock enabled (budget %llu MiB)\n",
                    (unsigned long long)mb);
        }
    }
    if (!enabled || len == 0 || g_wrap_mlock_suppress) return;
    for (uint32_t i = 0; i < n_ranges; i++) {
        if (ranges[i].off == offset && ranges[i].len == len) return;
    }
    if (n_ranges >= 4096) return;
    ranges[n_ranges].off = offset;
    ranges[n_ranges].len = len;
    n_ranges++;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t start = offset & ~(page - 1u);
    const uint64_t end = (offset + len + page - 1u) & ~(page - 1u);
    if (wired_bytes + (end - start) > budget_bytes) {
        if (!budget_warned) {
            budget_warned = 1;
            fprintf(stderr,
                    "ds4: backbone mlock budget exhausted at %.2f GiB (%u ranges); "
                    "remaining ranges stay evictable\n",
                    ds4_gpu_gib(wired_bytes), n_ranges);
        }
        return;
    }
    if (mlock((const uint8_t *)model_map + start, (size_t)(end - start)) == 0) {
        wired_bytes += end - start;
        if ((wired_bytes >> 29) > logged_half_gib) {   /* log every 512MiB */
            logged_half_gib = wired_bytes >> 29;
            fprintf(stderr, "ds4: backbone mlock wired %.2f GiB (%u ranges)\n",
                    ds4_gpu_gib(wired_bytes), n_ranges);
        }
    }
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) {
            ds4_gpu_backbone_mlock_register(model_map, offset, len);
            *inner_offset = offset - view_start;
            return g_model_views[i].buffer;
        }
    }

    fprintf(stderr,
            "ds4: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            ds4_gpu_gib(offset),
            ds4_gpu_gib(end));
    if (getenv("DS4_METAL_EXPERT_OFFLOAD_DEBUG") != NULL) {
        fprintf(stderr, "ds4:   wanted exact bytes [%llu, %llu) over %u views:\n",
                (unsigned long long)offset, (unsigned long long)end, g_model_view_count);
        for (uint32_t i = 0; i < g_model_view_count; i++) {
            const uint64_t vs = g_model_views[i].model_offset;
            const uint64_t ve = vs + g_model_views[i].bytes;
            if (offset < ve + (64ull << 20) && end + (64ull << 20) > vs) {
                fprintf(stderr, "ds4:     view[%u] bytes [%llu, %llu) resident=%d\n",
                        i, (unsigned long long)vs, (unsigned long long)ve,
                        g_model_views[i].resident_hint);
            }
        }
    }
    return nil;
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer score received undersized buffers\n");
            return 0;
        }

        if (n_head == 64 && head_dim == 128) {
            id<MTLComputePipelineState> direct_pipeline =
                ds4_gpu_hot_pipeline(g_dsv4_indexer_score_one_direct_pipeline,
                                        "kernel_dsv4_indexer_score_one_direct");
            if (!direct_pipeline) return 0;

            ds4_gpu_dsv4_indexer_scores_fused_args args = {
                .n_comp = n_comp,
                .n_tokens = 1,
                .n_head = n_head,
                .head_dim = head_dim,
                .pos0 = 0,
                .ratio = 4,
                .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
                .q_head_stride = (uint64_t)head_dim * sizeof(float),
                .weights_token_stride = (uint64_t)n_head * sizeof(float),
                .index_row_stride = (uint64_t)head_dim * sizeof(float),
                .score_token_stride = (uint64_t)n_comp * sizeof(float),
                .scale = scale,
            };

            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:direct_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
            [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
            [enc setThreadgroupMemoryLength:(128u + 4u) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_comp, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer direct score")) return 0;
            return 1;
        }

        const uint64_t head_score_bytes = (uint64_t)n_comp * n_head * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_head_scores_buffer,
                                             &g_indexer_head_scores_bytes,
                                             (NSUInteger)head_score_bytes,
                                             "ds4_indexer_head_scores")) {
            return 0;
        }

        ds4_gpu_q8_0_matvec_args dot_args =
            ds4_gpu_make_f32_mv_args(head_dim, n_comp, n_head);
        ds4_gpu_mv_dispatch dot_dispatch =
            ds4_gpu_make_plain_mv_dispatch(head_dim, 1);
        dot_args.nr0 = dot_dispatch.nr0;
        id<MTLComputePipelineState> dot_pipeline =
            ds4_gpu_get_mul_mv_pipeline(dot_dispatch.function_name, dot_dispatch.nsg);
        if (!dot_pipeline) return 0;
        ds4_gpu_dsv4_indexer_weighted_sum_args sum_args = {
            .ne00 = (int64_t)n_comp,
            .ne01 = 1,
            .ne02 = (int64_t)n_head,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * sizeof(float),
            .ne10 = (int64_t)n_head,
            .ne11 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)n_head * sizeof(float),
            .ne0 = (int64_t)n_comp,
            .ne1 = 1,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:dot_pipeline];
        [enc setBytes:&dot_args length:sizeof(dot_args) atIndex:0];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:3];
        if (dot_dispatch.smem) {
            [enc setThreadgroupMemoryLength:dot_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + (NSUInteger)dot_dispatch.nr0 - 1u) / (NSUInteger)dot_dispatch.nr0,
                                              n_head,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dot_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_weighted_sum_pipeline];
        [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer score")) return 0;
    }

    return 1;
}

static int ds4_gpu_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer prefill scores received undersized buffers\n");
            return 0;
        }
        if (head_dim != 128) {
            fprintf(stderr, "ds4: Metal fused DS4 indexer scores expect 128-wide rows\n");
            return 0;
        }
        /*
         * The NAX/TensorOps score builder is a prefill-only win.  At small
         * batches and in one-token decode the setup cost is not amortized, so
         * those paths keep the older direct/tiled score kernels.
         */
        const bool use_nax = ds4_gpu_mpp_available() && n_tokens >= 16u;
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(
            use_nax ? "kernel_dsv4_indexer_scores_nax" :
            (g_quality_mode ? "kernel_dsv4_indexer_scores_tiled_f32"
                            : "kernel_dsv4_indexer_scores_tiled"));
        if (!pipeline) return 0;

        ds4_gpu_dsv4_indexer_scores_fused_args args = {
            .n_comp = n_comp,
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .pos0 = pos0,
            .ratio = ratio,
            .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
            .q_head_stride = (uint64_t)head_dim * sizeof(float),
            .weights_token_stride = (uint64_t)n_head * sizeof(float),
            .index_row_stride = (uint64_t)head_dim * sizeof(float),
            .score_token_stride = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
        if (use_nax) {
            const NSUInteger q_shared = 16u * 32u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 16u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 15u) / 16u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (g_quality_mode) {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared + dot_shared) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer prefill scores")) return 0;
    }

    return 1;
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 pos0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 || top_k > n_comp) return 0;

    @autoreleasepool {
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        if (!scorebuf || !selbuf ||
            ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer top-k received undersized buffers\n");
            return 0;
        }
        NSUInteger max_threads = g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
        if (max_threads == 0) max_threads = 256;
        int32_t nth = 1;
        while ((uint32_t)nth < n_comp && (uint64_t)2u * (uint64_t)nth <= (uint64_t)max_threads) {
            nth *= 2;
        }
        const int32_t npr = (int32_t)((n_comp + (uint32_t)nth - 1u) / (uint32_t)nth);
        const int32_t block_top_k = (int32_t)(top_k < (uint32_t)nth ? top_k : (uint32_t)nth);
        int32_t work_width = (int32_t)top_k;
        if (npr > 1) {
            const int32_t last_block = (int32_t)n_comp - (npr - 1) * nth;
            work_width = (npr - 1) * block_top_k + (last_block < block_top_k ? last_block : block_top_k);
        }
        const uint64_t scratch_row_bytes = (uint64_t)work_width * sizeof(uint32_t);
        const bool one_pass = npr <= 1;
        const uint64_t scratch_bytes = one_pass ? scratch_row_bytes * n_tokens :
            2u * scratch_row_bytes * n_tokens;
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_topk_buffer,
                                             &g_indexer_topk_bytes,
                                             (NSUInteger)scratch_bytes,
                                             "ds4_indexer_topk")) {
            return 0;
        }

        ds4_gpu_kargs_argsort args = {
            .ne00 = (int32_t)n_comp,
            .ne01 = (int32_t)n_tokens,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .ne0 = work_width,
            .ne1 = (int32_t)n_tokens,
            .ne2 = 1,
            .ne3 = 1,
            .top_k = block_top_k,
        };
        const NSUInteger smem = (((NSUInteger)nth * sizeof(int32_t)) + 15u) & ~(NSUInteger)15u;

        NSUInteger cur_off = 0;
        NSUInteger next_off = (NSUInteger)scratch_row_bytes * n_tokens;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_argsort_f32_i32_desc_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
        [enc setBuffer:one_pass ? selbuf : g_indexer_topk_buffer
              offset:one_pass ? ds4_gpu_tensor_offset(selected) : cur_off
             atIndex:2];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)npr * n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        int32_t len = block_top_k;
        while (len < work_width) {
            const int32_t nm = (work_width + 2 * len - 1) / (2 * len);
            const bool final_merge = nm == 1;
            NSUInteger merge_threads = g_argsort_merge_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
            if (merge_threads == 0 || merge_threads > 512u) merge_threads = 512u;
            if (merge_threads > (NSUInteger)len) merge_threads = (NSUInteger)len;
            if (merge_threads == 0) merge_threads = 1;

            ds4_gpu_kargs_argsort_merge merge_args = {
                .ne00 = (int64_t)n_comp,
                .ne01 = (int64_t)n_tokens,
                .ne02 = 1,
                .ne03 = 1,
                .nb00 = sizeof(float),
                .nb01 = (uint64_t)n_comp * sizeof(float),
                .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .ne0 = work_width,
                .ne1 = (int32_t)n_tokens,
                .ne2 = 1,
                .ne3 = 1,
                .top_k = nm == 1 ? (int32_t)top_k : work_width,
                .len = len,
            };

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:g_argsort_merge_f32_i32_desc_pipeline];
            [enc setBytes:&merge_args length:sizeof(merge_args) atIndex:0];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
            [enc setBuffer:g_indexer_topk_buffer offset:cur_off atIndex:2];
            [enc setBuffer:final_merge ? selbuf : g_indexer_topk_buffer
                  offset:final_merge ? ds4_gpu_tensor_offset(selected) : next_off
                 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nm * n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(merge_threads, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            const NSUInteger tmp = cur_off;
            cur_off = next_off;
            next_off = tmp;
            len <<= 1;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer top-k")) return 0;
    }

    return 1;
}

int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0) return 0;
    if (ds4_gpu_tensor_bytes(out_idx) < sizeof(int32_t) ||
        ds4_gpu_tensor_bytes(logits) < (uint64_t)n_vocab * sizeof(float)) {
        fprintf(stderr, "ds4: Metal graph argmax received undersized buffers\n");
        return 0;
    }

    return ds4_gpu_indexer_topk_tensor(out_idx, logits, n_vocab, 1, 1);
}

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;

    @autoreleasepool {
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        const uint64_t mask_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> maskbuf = ds4_gpu_tensor_buffer(mask);
        if (!topkbuf || !maskbuf ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(mask) < mask_bytes) {
            fprintf(stderr, "ds4: Metal dsv4 top-k mask received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)n_comp,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)n_comp * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_scatter_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)top_k * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "dsv4 top-k mask")) return 0;
    }

    return 1;
}

static int ds4_gpu_matmul_q8_0_legacy_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
            if (out_dim > 65536u) mv_dispatch.nsg = 8;
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matvec")) {
                return 0;
            }
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(1, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor mul_mv_ext")) {
                return 0;
            }
            return 1;
        }

        /*
         * Dense Q8_0 prefill is the cleanest DS4 TensorOps shape: M/N/K are
         * aligned and the RHS activation matrix is already dense.  The retained
         * kernel dequantizes each 64x32 weight tile to half in threadgroup
         * memory, then uses direct-RHS MPP for the activation tile.  This avoids
         * staging RHS into threadgroup memory and was the direct replacement for
         * the slower generic MPP prototype.
         */
        if (ds4_gpu_mpp_available() &&
            n_tok >= 32u &&
            (in_dim % 64u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64"
                    : "kernel_mul_mm_q8_0_f32_nax_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 NAX tensor matmul")) {
                    return 0;
                }
                return 1;
            }
            ds4_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_q8_0_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matmul")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    const int profile_requested =
        n_tok > 8u && ds4_gpu_env_bool("DS4_METAL_Q8_PREFILL_PROFILE") > 0;
    int profile_prefill = 0;
    int split_batch_for_profile = 0;
    const char *profile_label = NULL;
    char profile_label_buf[128];
    char profile_fallback[128];
    if (profile_requested) {
        snprintf(profile_fallback, sizeof(profile_fallback),
                 "q8 weight_off=%llu in=%llu out=%llu tok=%llu",
                 (unsigned long long)weight_offset,
                 (unsigned long long)in_dim,
                 (unsigned long long)out_dim,
                 (unsigned long long)n_tok);
        snprintf(profile_label_buf, sizeof(profile_label_buf), "%s", profile_fallback);
        profile_label = profile_label_buf;
        const char *profile_filter = getenv("DS4_METAL_Q8_PREFILL_PROFILE_FILTER");
        profile_prefill =
            profile_requested &&
            (!profile_filter || !profile_filter[0] ||
             strstr(profile_label, profile_filter) != NULL);
    }
    if (profile_prefill) {
        if (g_batch_cb) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            split_batch_for_profile = 1;
        }
    }

    const double profile_t0 = profile_prefill ? ds4_gpu_now_ms() : 0.0;
    int ok = ds4_gpu_matmul_q8_0_legacy_tensor(out, model_map, model_size,
                                                weight_offset, in_dim, out_dim,
                                                x, n_tok);
    if (profile_prefill) {
        if (split_batch_for_profile && ds4_gpu_end_commands() == 0) {
            ok = 0;
        }
        const double elapsed_ms = ds4_gpu_now_ms() - profile_t0;
        fprintf(stderr,
                "ds4: Metal Q8_0 prefill profile %s in=%llu out=%llu tok=%llu %.3f ms\n",
                profile_label ? profile_label : profile_fallback,
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (unsigned long long)n_tok,
                elapsed_ms);
        if (split_batch_for_profile && ds4_gpu_begin_commands() == 0) {
            ok = 0;
        }
    }
    return ok;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        !isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < out_bytes ||
            ds4_gpu_tensor_bytes(up) < out_bytes ||
            ds4_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        ds4_gpu_q8_0_matvec_args args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_shared_gate_up_swiglu_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:4];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                                  (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "shared expert fused gate/up")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F16 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch =
                ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
            if (!g_quality_mode && (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
                mv_dispatch.nr0 = 4;
                mv_dispatch.smem = 32u * 4u * sizeof(float);
            }
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor matvec")) return 0;
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(0, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, sizeof(uint16_t), row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor mul_mv_ext")) return 0;
            return 1;
        }

        /*
         * Same direct-RHS TensorOps structure as Q8_0, but for F16 model
         * matrices.  The 128-token RHS tile is kept when the batch alignment
         * allows it because the later tile_n=64 retest was neutral/slower.
         */
        if (ds4_gpu_mpp_available() &&
            n_tok >= 32u &&
            (in_dim % 32u) == 0 &&
            (out_dim % 64u) == 0 &&
            (n_tok % 32u) == 0) {
            uint64_t nax_tile_n = 32u;
            if ((n_tok % 128u) == 0) {
                nax_tile_n = 128u;
            } else if ((n_tok % 64u) == 0) {
                nax_tile_n = 64u;
            }
            const char *nax_fn = nax_tile_n == 128u
                ? "kernel_mul_mm_f16_f32_mpp_direct_rhs_n128"
                : (nax_tile_n == 64u
                    ? "kernel_mul_mm_f16_f32_mpp_direct_rhs_n64"
                    : "kernel_mul_mm_f16_f32_mpp_direct_rhs");
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mm_pipeline(nax_fn, false, false);
            if (pipeline) {
                ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

                id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
                [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
                [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
                [enc setThreadgroupMemoryLength:64u * 32u * sizeof(uint16_t) atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(n_tok / nax_tile_n),
                                                      (NSUInteger)out_dim / 64u,
                                                      1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ds4_gpu_end_compute_encoder(cb, enc);

                if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 NAX tensor matmul")) {
                    return 0;
                }
                return 1;
            }
            ds4_gpu_warn_mpp_fallback();
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_f16_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor matmul")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok != 1 || (in_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outabuf = ds4_gpu_tensor_buffer(out_a);
        id<MTLBuffer> outbbuf = ds4_gpu_tensor_buffer(out_b);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outabuf || !outbbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out_a) < out_bytes ||
            ds4_gpu_tensor_bytes(out_b) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 paired matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_a_offset > model_size || weight_bytes > model_size - weight_a_offset ||
            weight_b_offset > model_size || weight_bytes > model_size - weight_b_offset) {
            fprintf(stderr, "ds4: Metal F16 paired matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_a = 0;
        uint64_t inner_b = 0;
        id<MTLBuffer> wabuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_a_offset, weight_bytes,
                                                         &inner_a);
        id<MTLBuffer> wbbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_b_offset, weight_bytes,
                                                         &inner_b);
        if (!wabuf || !wbbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
        if (ds4_gpu_use_compressor_pair_nr4() &&
            (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
            mv_dispatch.nr0 = 4;
            mv_dispatch.smem = 32u * 4u * sizeof(float);
        }
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_f16_f32_pair_4", mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wabuf offset:(NSUInteger)inner_a atIndex:1];
        [enc setBuffer:wbbuf offset:(NSUInteger)inner_b atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outabuf offset:ds4_gpu_tensor_offset(out_a) atIndex:4];
        [enc setBuffer:outbbuf offset:ds4_gpu_tensor_offset(out_b) atIndex:5];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 paired matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX || n_tok != 1) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(float);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_f32_mv_args(in_dim, out_dim, 1);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 1);
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F32 tensor matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> rowbuf = ds4_gpu_tensor_buffer(row);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_bytes = row_bytes * n_hc;
        if (!rowbuf || !outbuf ||
            ds4_gpu_tensor_bytes(row) < row_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal HC repeat received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  rowbuf,
                                                  ds4_gpu_tensor_offset(row),
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC repeat")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal plain RMS norm received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "plain RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        const uint64_t bytes = row_bytes * rows;
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal weighted RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal weighted RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, row_bytes, &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "weighted RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_out || !kv || q_n == 0 || kv_n == 0 || rows == 0 ||
        (q_n & 3u) != 0 || (kv_n & 3u) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> kvoutbuf = ds4_gpu_tensor_buffer(kv_out);

        const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
        const uint64_t kv_row_bytes = (uint64_t)kv_n * sizeof(float);
        if (!qbuf || !qoutbuf || !kvbuf || !kvoutbuf ||
            ds4_gpu_tensor_bytes(q) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(q_out) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv) < kv_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv_out) < kv_row_bytes * rows) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
            kv_weight_offset > model_size || kv_row_bytes > model_size - kv_weight_offset) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm weight range is outside the mapped model\n");
            return 0;
        }

        uint64_t q_inner_offset = 0;
        uint64_t kv_inner_offset = 0;
        id<MTLBuffer> q_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                          q_weight_offset, q_row_bytes,
                                                          &q_inner_offset);
        if (!q_wbuf) return 0;
        id<MTLBuffer> kv_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           kv_weight_offset, kv_row_bytes,
                                                           &kv_inner_offset);
        if (!kv_wbuf) return 0;

        ds4_gpu_qkv_rms_norm_args args = {
            .q_n = (int32_t)q_n,
            .q_n4 = (int32_t)(q_n / 4u),
            .kv_n = (int32_t)kv_n,
            .kv_n4 = (int32_t)(kv_n / 4u),
            .q_row_stride = q_row_bytes,
            .kv_row_stride = kv_row_bytes,
            .eps = eps,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_qkv_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:q_wbuf offset:(NSUInteger)q_inner_offset atIndex:2];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:3];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:4];
        [enc setBuffer:kv_wbuf offset:(NSUInteger)kv_inner_offset atIndex:5];
        [enc setBuffer:kvoutbuf offset:ds4_gpu_tensor_offset(kv_out) atIndex:6];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(q_n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "fused q/kv RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || (head_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal head RMS norm received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_3d_args(head_dim, n_head, n_tok, eps);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_pipeline_threads(head_dim, g_rms_norm_plain_pipeline), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "head RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u) != 0) {
        return 0;
    }
    if (n_rot == 0) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal RoPE received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rope_tail_batch_args args = ds4_gpu_make_rope_tail_args(
            n_tok, n_head, head_dim, n_rot, n_ctx_orig, inverse,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_rope_tail_inplace(cb,
                                                xbuf,
                                                ds4_gpu_tensor_offset(x),
                                                &args,
                                                n_tok,
                                                n_head,
                                                head_dim,
                                                pos0,
                                                1)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "RoPE tail")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (n_rot == head_dim) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 FP8 KV quantize received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_fp8_kv_quantize_args args = {
            .ne00 = head_dim,
            .ne01 = n_tok,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)head_dim * sizeof(float),
            .nb02 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb03 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)head_dim * sizeof(float),
            .nb2 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb3 = (uint64_t)n_tok * head_dim * sizeof(float),
            .n_rot = (int32_t)n_rot,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_fp8_kv_quantize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 FP8 KV quantize")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_rows == 0 || head_dim != 128u) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_rows * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 indexer QAT received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_indexer_qat_args args = {
            .n_rows = n_rows,
            .head_dim = head_dim,
            .row_stride = (uint64_t)head_dim * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_qat_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 indexer Hadamard+FP4")) return 0;
    }

    return 1;
}

static void ds4_gpu_set_rows_thread_shape(
        uint32_t    width,
        NSUInteger *nth_out,
        NSUInteger *nrptg_out) {
    const NSUInteger nk0 = width ? (NSUInteger)width : 1u;
    const NSUInteger max_threads = g_set_rows_f32_i32_pipeline
        ? (NSUInteger)g_set_rows_f32_i32_pipeline.maxTotalThreadsPerThreadgroup
        : 1024u;

    NSUInteger nth = 32u;
    while (nth < nk0 && nth < max_threads) {
        nth *= 2u;
    }

    NSUInteger nrptg = 1u;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1u) / nk0;
        nth = nk0;
        if (nrptg * nth > max_threads) {
            nrptg--;
        }
    }

    if (nth > nk0) nth = nk0;
    if (nth == 0u) nth = 1u;
    if (nrptg == 0u) nrptg = 1u;

    *nth_out = nth;
    *nrptg_out = nrptg;
}

static int ds4_gpu_encode_f16_round_copy_for_raw_store(
        id<MTLCommandBuffer>   cb,
        const ds4_gpu_tensor *src,
        uint32_t               n) {
    id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
    const uint64_t src_bytes = (uint64_t)n * sizeof(float);
    if (!srcbuf || ds4_gpu_tensor_bytes(src) < src_bytes) {
        fprintf(stderr, "ds4: Metal raw KV store received undersized source buffer\n");
        return 0;
    }
    if (!ds4_gpu_ensure_scratch_buffer(&g_f16_round_scratch_buffer,
                                         &g_f16_round_scratch_bytes,
                                         (NSUInteger)n * sizeof(uint16_t),
                                         "ds4_f16_round_scratch") ||
        !ds4_gpu_ensure_scratch_buffer(&g_raw_store_round_buffer,
                                         &g_raw_store_round_bytes,
                                         (NSUInteger)n * sizeof(float),
                                         "ds4_raw_store_round")) {
        return 0;
    }

    ds4_gpu_cpy_args f32_to_f16 =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
    ds4_gpu_cpy_args f16_to_f32 =
        ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
    const NSUInteger nth_f32_f16 = ds4_gpu_cpy_threads(n, g_cpy_f32_f16_pipeline);
    const NSUInteger nth_f16_f32 = ds4_gpu_cpy_threads(n, g_cpy_f16_f32_pipeline);
    const NSUInteger groups_f32_f16 = ((NSUInteger)n + nth_f32_f16 - 1u) / nth_f32_f16;
    const NSUInteger groups_f16_f32 = ((NSUInteger)n + nth_f16_f32 - 1u) / nth_f16_f32;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&f32_to_f16 length:sizeof(f32_to_f16) atIndex:0];
    [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
    [enc setBuffer:g_f16_round_scratch_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_f32_f16, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth_f32_f16, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f32_pipeline];
    [enc setBytes:&f16_to_f32 length:sizeof(f16_to_f32) atIndex:0];
    [enc setBuffer:g_f16_round_scratch_buffer offset:0 atIndex:1];
    [enc setBuffer:g_raw_store_round_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_f16_f32, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth_f16_f32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_set_rows_f32_i32(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *dst,
        id<MTLBuffer>        srcbuf,
        NSUInteger           src_off,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             dst_rows,
        uint32_t             width) {
    id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
    const uint64_t dst_bytes = (uint64_t)dst_rows * width * sizeof(float);
    const uint64_t src_bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!dstbuf || !srcbuf || !rows || n_rows == 0 || width == 0 ||
        ds4_gpu_tensor_bytes(dst) < dst_bytes ||
        src_bytes > NSUIntegerMax - src_off) {
        fprintf(stderr, "ds4: Metal DS4 set_rows received invalid buffers\n");
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rows_bytes = (uint64_t)n_rows * sizeof(int32_t);
    ds4_gpu_set_rows_args args = {
        .nk0 = (int32_t)width,
        .ne01 = (int32_t)n_rows,
        .nb01 = row_bytes,
        .nb02 = (uint64_t)n_rows * row_bytes,
        .nb03 = (uint64_t)n_rows * row_bytes,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(int32_t),
        .nb11 = rows_bytes,
        .nb12 = rows_bytes,
        .nb1 = row_bytes,
        .nb2 = (uint64_t)dst_rows * row_bytes,
        .nb3 = (uint64_t)dst_rows * row_bytes,
    };

    NSUInteger nth;
    NSUInteger nrptg;
    ds4_gpu_set_rows_thread_shape(width, &nth, &nrptg);

    id<MTLBuffer> rowsbuf = nil;
    if (rows_bytes > 4096u) {
        rowsbuf = ds4_gpu_new_transient_buffer((NSUInteger)rows_bytes, "ds4_set_rows_indices");
        if (!rowsbuf) return 0;
        memcpy([rowsbuf contents], rows, (NSUInteger)rows_bytes);
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_set_rows_f32_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:srcbuf offset:src_off atIndex:1];
    if (rowsbuf) {
        [enc setBuffer:rowsbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:rows length:(NSUInteger)rows_bytes atIndex:2];
    }
    [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(dst) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_rows + nrptg - 1u) / nrptg, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, nrptg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !a || !b || !out || n == 0) return 0;

    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    ds4_gpu_bin_args args = {
        .ne00 = (int32_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = row_bytes,
        .nb13 = row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };

    NSUInteger nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)n && nth < nth_max) {
        nth *= 2u;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_add_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV store received undersized destination buffer\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t row_i32 = (int32_t)row;
        if (!ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, head_dim) ||
            !ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               &row_i32,
                                               1,
                                               raw_cap,
                                               head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV DS4 set_rows store")) return 0;
    }

    return 1;
}

/* Release decode fused KV finalizer.  Reference paths are selected by the C
 * graph driver; this Objective-C entry point always means "use the fused
 * Metal kernel." */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !raw_cache || raw_cap == 0 || row >= raw_cap || head_dim == 0 ||
        n_rot > head_dim || raw_cap > INT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_cache);
        const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (!kvbuf || !rawbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal fused KV FP8/raw-store received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_kv_fp8_store_args args = {
            .head_dim = (int32_t)head_dim,
            .n_rot = (int32_t)n_rot,
            .raw_row = (int32_t)row,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_kv_fp8_store_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_cache) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "KV FP8/raw-store fused")) return 0;
    }

    return 1;
}

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV batch store received undersized destination buffer\n");
            return 0;
        }

        int32_t rows_stack[512];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate raw KV set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            rows[t] = (int32_t)((pos0 + t) % raw_cap);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        const uint64_t n = (uint64_t)n_tokens * head_dim;
        const int ok = n <= UINT32_MAX &&
            ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, (uint32_t)n) &&
            ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               rows,
                                               n_tokens,
                                               raw_cap,
                                               head_dim);
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV batch DS4 set_rows store")) return 0;
    }

    return 1;
}

static int ds4_gpu_encode_compressor_score_with_ape(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        score_src,
        NSUInteger           score_src_offset,
        id<MTLBuffer>        score_dst,
        NSUInteger           score_dst_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        uint32_t             n_tokens) {
    if (!cb || !score_src || !score_dst || !apebuf ||
        width == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    const uint64_t total_elems64 = (uint64_t)n_tokens * width;
    if (total_elems64 > UINT32_MAX) {
        fprintf(stderr, "ds4: Metal compressor APE add received too many elements\n");
        return 0;
    }
    const uint32_t total_elems = (uint32_t)total_elems64;
    const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                         &g_compressor_store_ape_bytes,
                                         scratch_bytes,
                                         "ds4_compressor_store_ape")) {
        return 0;
    }

    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    uint32_t copied_rows = 0;
    uint32_t pos_mod = pos0 % ratio;
    while (copied_rows < n_tokens) {
        uint32_t seg_rows = ratio - pos_mod;
        if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
        const uint32_t seg_elems = seg_rows * width;
        const NSUInteger src_off = ape_offset + (NSUInteger)pos_mod * width * elem_ape;
        const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
        int ok;
        if (ape_type == 1u) {
            ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        } else {
            ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        }
        if (!ok) return 0;
        copied_rows += seg_rows;
        pos_mod = 0;
    }

    return ds4_gpu_encode_add_f32_1d(cb,
                                       score_src,
                                       score_src_offset,
                                       g_compressor_store_ape_buffer,
                                       0,
                                       score_dst,
                                       score_dst_offset,
                                       total_elems);
}

static int ds4_gpu_encode_compressor_set_rows_projected(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             state_rows) {
    if (!cb || !state_kv || !state_score || !kvbuf || !scorebuf ||
        !apebuf || !rows || width == 0 || n_rows == 0 || state_rows == 0) {
        return 0;
    }

    const NSUInteger score_scratch_bytes = (NSUInteger)n_rows * width * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                         &g_compressor_store_score_bytes,
                                         score_scratch_bytes,
                                         "ds4_compressor_store_score")) {
        return 0;
    }

    return ds4_gpu_encode_compressor_score_with_ape(cb,
                                                      scorebuf,
                                                      score_offset,
                                                      g_compressor_store_score_buffer,
                                                      0,
                                                      apebuf,
                                                      ape_offset,
                                                      ape_type,
                                                      width,
                                                      ratio,
                                                      pos0,
                                                      n_rows) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_kv,
                                             kvbuf,
                                             kv_offset,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_score,
                                             g_compressor_store_score_buffer,
                                             0,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width);
}

static int ds4_gpu_compressor_store_one_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                width,
        uint32_t                ratio,
        uint32_t                pos) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        width == 0 || ratio == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_hot_pipeline(g_dsv4_compressor_store_one_pipeline,
                                "kernel_dsv4_compressor_store_one");
    if (!pipeline) return 0;

    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        ds4_gpu_tensor_bytes(kv) < row_bytes ||
        ds4_gpu_tensor_bytes(sc) < row_bytes ||
        ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    uint64_t ape_inner = 0;
    id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                       ape_offset, ape_bytes,
                                                       &ape_inner);
    id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
    id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!apebuf || !kvbuf || !scbuf || !statekvbuf || !statescbuf) return 0;

    ds4_gpu_dsv4_compressor_store_one_args args = {
        .width = width,
        .ratio = ratio,
        .pos = pos,
        .ape_type = ape_type,
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;

    const NSUInteger nth = 256u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
    [enc setBuffer:scbuf offset:ds4_gpu_tensor_offset(sc) atIndex:2];
    [enc setBuffer:apebuf offset:(NSUInteger)ape_inner atIndex:3];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:4];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)width + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return ds4_gpu_finish_command_buffer(cb, owned, "compressor one-row store");
}

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor batch APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        if (!kvbuf || !scbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor batch store received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const uint64_t total_elems64 = (uint64_t)n_tokens * width;
        if (total_elems64 > UINT32_MAX || state_rows > INT32_MAX) {
            fprintf(stderr, "ds4: Metal compressor batch store received too many elements\n");
            return 0;
        }
        const uint32_t total_elems = (uint32_t)total_elems64;
        const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                             &g_compressor_store_ape_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_ape") ||
            !ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                             &g_compressor_store_score_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_score")) {
            return 0;
        }

        int32_t rows_stack[16];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate compressor set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t pos_mod = (pos0 + t) % ratio;
            rows[t] = (int32_t)(ratio == 4u ? ratio + pos_mod : pos_mod);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        int ok = 1;
        uint32_t copied_rows = 0;
        uint32_t pos_mod = pos0 % ratio;
        while (ok && copied_rows < n_tokens) {
            uint32_t seg_rows = ratio - pos_mod;
            if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
            const uint32_t seg_elems = seg_rows * width;
            const NSUInteger src_off = (NSUInteger)ape_inner +
                                       (NSUInteger)pos_mod * width * elem_ape;
            const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
            if (ape_type == 1u) {
                ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            } else {
                ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            }
            copied_rows += seg_rows;
            pos_mod = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_add_f32_1d(cb,
                                             scbuf,
                                             ds4_gpu_tensor_offset(sc),
                                             g_compressor_store_ape_buffer,
                                             0,
                                             g_compressor_store_score_buffer,
                                             0,
                                             total_elems);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_kv,
                                                   kvbuf,
                                                   ds4_gpu_tensor_offset(kv),
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_score,
                                                   g_compressor_store_score_buffer,
                                                   0,
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "compressor batch DS4 store")) return 0;
    }

    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_contiguous_3d_args(
        uint32_t cols,
        uint32_t rows,
        uint32_t planes) {
    const uint64_t row_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)rows,
        .ne12 = (int32_t)planes,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = (int32_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_softmax_f32_contiguous(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows,
        uint32_t             planes) {
    if (!cb || !src || !dst || width == 0 || rows == 0 || planes == 0) return 0;

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    ds4_gpu_softmax_args args = {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne11 = (int32_t)width,
        .ne12 = (int32_t)rows,
        .ne13 = (int32_t)planes,
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .scale = 1.0f,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 1,
    };

    id<MTLComputePipelineState> pipeline =
        (width % 4u) == 0 ? g_soft_max_f32_4_pipeline : g_soft_max_f32_pipeline;
    if (!pipeline) return 0;

    NSUInteger nth = 32u;
    if ((width % 4u) == 0) {
        while (nth < (NSUInteger)(width / 4u) &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    } else {
        while (nth < (NSUInteger)width &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    }
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:src offset:src_off atIndex:2];
    [enc setBuffer:src offset:src_off atIndex:3];
    [enc setBuffer:dst offset:dst_off atIndex:4];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf || n_rows == 0 || head_dim == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * sizeof(float)) {
        return 0;
    }

    const NSUInteger pack_bytes = (NSUInteger)n_rows * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_product_buffer,
                                         &g_compressor_pool_product_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_product") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_cont_buffer,
                                         &g_compressor_pool_score_cont_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_score_cont") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_softmax_buffer,
                                         &g_compressor_pool_softmax_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_softmax")) {
        return 0;
    }

    const uint64_t cont_row_stride = (uint64_t)n_rows * sizeof(float);
    const uint64_t cont_plane_stride = (uint64_t)head_dim * cont_row_stride;

    /*
     * Keep the n_comp == 1 compressor path as the unfused graph sequence:
     *
     *   score = soft_max(contiguous(score))
     *   pooled = sum_rows(contiguous(kv) * score)
     *
     * The fused DS4 pool kernel is mathematically equivalent, but it reduces in
     * a different order. That is enough to create ~1e-6 compressor differences
     * and later FP8/routing flips, so this path intentionally keeps the same
     * operation boundary and memory layout as the graph.
     */
    ds4_gpu_bin_args mul_args =
        ds4_gpu_make_bin_contiguous_3d_args(n_rows, head_dim, 1);

    return
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    kvbuf,
                                                    kv_offset,
                                                    g_compressor_pool_product_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    kv_nb0,
                                                    kv_nb1,
                                                    kv_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    scorebuf,
                                                    score_offset,
                                                    g_compressor_pool_score_cont_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    score_nb0,
                                                    score_nb1,
                                                    score_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_softmax_f32_contiguous(cb,
                                                g_compressor_pool_score_cont_buffer,
                                                0,
                                                g_compressor_pool_softmax_buffer,
                                                0,
                                                n_rows,
                                                head_dim,
                                                1) &&
        ds4_gpu_encode_bin_f32_rows(cb,
                                      g_mul_pipeline,
                                      &mul_args,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      g_compressor_pool_softmax_buffer,
                                      0,
                                      g_compressor_pool_product_buffer,
                                      0) &&
        ds4_gpu_encode_sum_rows_f32(cb,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      outbuf,
                                      ds4_gpu_tensor_offset(out),
                                      n_rows,
                                      head_dim);
}

static int ds4_gpu_encode_dsv4_softmax_pool(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim,
        uint32_t             n_comp) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf ||
        n_rows == 0 || head_dim == 0 || n_comp == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * n_comp * sizeof(float)) {
        return 0;
    }

    if (n_comp == 1) {
        return ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(cb,
                                                                out,
                                                                kvbuf,
                                                                kv_offset,
                                                                kv_nb0,
                                                                kv_nb1,
                                                                kv_nb2,
                                                                scorebuf,
                                                                score_offset,
                                                                score_nb0,
                                                                score_nb1,
                                                                score_nb2,
                                                                n_rows,
                                                                head_dim);
    }

    ds4_gpu_dsv4_softmax_pool_args args = {
        .ne00 = (int64_t)n_rows,
        .ne01 = (int64_t)head_dim,
        .ne02 = (int64_t)n_comp,
        .nb00 = kv_nb0,
        .nb01 = kv_nb1,
        .nb02 = kv_nb2,
        .nb10 = score_nb0,
        .nb11 = score_nb1,
        .nb12 = score_nb2,
        .ne0 = (int64_t)head_dim,
        .ne1 = (int64_t)n_comp,
        .nb0 = sizeof(float),
        .nb1 = (uint64_t)head_dim * sizeof(float),
    };
    const uint64_t n = (uint64_t)head_dim * n_comp;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_softmax_pool_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:kv_offset atIndex:1];
    [enc setBuffer:scorebuf offset:score_offset atIndex:2];
    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_concat_f32_dim1(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src0,
        NSUInteger           src0_offset,
        uint32_t             src0_rows,
        uint64_t             src0_row_stride,
        id<MTLBuffer>        src1,
        NSUInteger           src1_offset,
        uint32_t             src1_rows,
        uint64_t             src1_row_stride,
        id<MTLBuffer>        dst,
        NSUInteger           dst_offset,
        uint32_t             cols,
        uint64_t             dst_row_stride) {
    if (!cb || !src0 || !src1 || !dst || cols == 0 || src0_rows == 0 || src1_rows == 0) {
        return 0;
    }

    const uint32_t rows = src0_rows + src1_rows;
    const uint64_t src0_plane = (uint64_t)src0_rows * src0_row_stride;
    const uint64_t src1_plane = (uint64_t)src1_rows * src1_row_stride;
    const uint64_t dst_plane = (uint64_t)rows * dst_row_stride;
    ds4_gpu_concat_args args = {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_row_stride,
        .nb02 = src0_plane,
        .nb03 = src0_plane,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)src1_rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_stride,
        .nb12 = src1_plane,
        .nb13 = src1_plane,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane,
        .nb3 = dst_plane,
        .dim = 1,
    };

    NSUInteger nth = cols < 1024u ? (NSUInteger)cols : 1024u;
    const NSUInteger max_threads = g_concat_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_concat_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src0 offset:src0_offset atIndex:1];
    [enc setBuffer:src1 offset:src1_offset atIndex:2];
    [enc setBuffer:dst offset:dst_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_compressor_pool(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *out,
        const ds4_gpu_tensor *state_kv,
        const ds4_gpu_tensor *state_score,
        uint32_t               head_dim,
        uint32_t               ratio) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !out || !statekvbuf || !statescbuf || head_dim == 0 || ratio == 0) return 0;

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rows = coff * ratio;
    const uint64_t state_bytes = (uint64_t)width * rows * sizeof(float);
    if (ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    if (ratio != 4u) {
        const uint64_t row_stride = (uint64_t)width * sizeof(float);
        return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                  out,
                                                  statekvbuf,
                                                  ds4_gpu_tensor_offset(state_kv),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  statescbuf,
                                                  ds4_gpu_tensor_offset(state_score),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  ratio,
                                                  head_dim,
                                                  1);
    }

    const NSUInteger packed_bytes = (NSUInteger)8u * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                         &g_compressor_pool_kv_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                         &g_compressor_pool_score_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_score")) {
        return 0;
    }

    const uint64_t state_row_stride = (uint64_t)width * sizeof(float);
    const uint64_t pool_row_stride = (uint64_t)head_dim * sizeof(float);
    const NSUInteger curr_offset = (NSUInteger)4u * state_row_stride +
                                   (NSUInteger)head_dim * sizeof(float);
    if (!ds4_gpu_encode_concat_f32_dim1(cb,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv),
                                          4,
                                          state_row_stride,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_kv_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride) ||
        !ds4_gpu_encode_concat_f32_dim1(cb,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score),
                                          4,
                                          state_row_stride,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_score_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride)) {
        return 0;
    }

    return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                              out,
                                              g_compressor_pool_kv_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              g_compressor_pool_score_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              8,
                                              head_dim,
                                              1);
}

static int ds4_gpu_encode_compressor_shift_ratio4(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        uint32_t             width) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !statekvbuf || !statescbuf || !g_dsv4_ratio4_shift_pipeline || width == 0) return 0;

    ds4_gpu_dsv4_ratio4_shift_args args = { .width = width };
    const uint32_t n = 4u * width;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_ratio4_shift_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:1];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_comp = n_tokens / ratio;
        const uint32_t cutoff = n_comp * ratio;
        const uint32_t rem = n_tokens - cutoff;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor prefill received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }

        if (ok && ratio == 4u) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const int have_prev = cutoff >= ratio ? 1 : 0;
            const uint32_t prev_start = rem == 0 ? cutoff - ratio : cutoff - ratio;
            if (have_prev) {
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)prev_start * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)prev_start * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + prev_start,
                                                                     rows_prev,
                                                                     4,
                                                                     state_rows);
            }
            if (ok && rem != 0) {
                int32_t rows_cur[4];
                for (uint32_t i = 0; i < rem; i++) rows_cur[i] = (int32_t)(ratio + i);
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + cutoff,
                                                                     rows_cur,
                                                                     rem,
                                                                     state_rows);
            }
        } else if (ok && rem != 0) {
            int32_t rows[128];
            if (rem > (uint32_t)(sizeof(rows) / sizeof(rows[0]))) {
                fprintf(stderr, "ds4: Metal compressor prefill remainder exceeds local row list\n");
                ok = 0;
            } else {
                for (uint32_t i = 0; i < rem; i++) rows[i] = (int32_t)i;
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + cutoff,
                                                                     rows,
                                                                     rem,
                                                                     state_rows);
            }
        }

        if (ok && n_comp != 0) {
            const NSUInteger score_bytes = (NSUInteger)cutoff * width * sizeof(float);
            if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                 &g_compressor_store_score_bytes,
                                                 score_bytes,
                                                 "ds4_compressor_store_score")) {
                ok = 0;
            }
            if (ok) {
                ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc),
                                                                 g_compressor_store_score_buffer,
                                                                 0,
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 cutoff);
            }

            if (ok && ratio == 4u) {
                const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
                if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                     &g_compressor_pool_kv_bytes,
                                                     pack_bytes,
                                                     "ds4_compressor_pool_kv") ||
                    !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                     &g_compressor_pool_score_bytes,
                                                     pack_bytes,
                                                     "ds4_compressor_pool_score")) {
                    ok = 0;
                }
                if (ok) {
                    ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                        g_compressor_pool_kv_buffer,
                                                        0,
                                                        head_dim,
                                                        8u * n_comp,
                                                        0.0f) &&
                         ds4_gpu_encode_fill_f32_rows(cb,
                                                        g_compressor_pool_score_buffer,
                                                        0,
                                                        head_dim,
                                                        8u * n_comp,
                                                        ds4_gpu_negative_infinity());
                }
                if (ok) {
                    const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
                    const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
                    const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
                    const uint64_t dst_plane_stride = 8ull * dst_row_stride;
                    ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         kvbuf,
                                                         ds4_gpu_tensor_offset(kv) +
                                                                 (NSUInteger)head_dim * sizeof(float),
                                                         g_compressor_pool_kv_buffer,
                                                         (NSUInteger)4u * head_dim * sizeof(float),
                                                         head_dim,
                                                         ratio,
                                                         n_comp,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride) &&
                         ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         g_compressor_store_score_buffer,
                                                         (NSUInteger)head_dim * sizeof(float),
                                                         g_compressor_pool_score_buffer,
                                                         (NSUInteger)4u * head_dim * sizeof(float),
                                                         head_dim,
                                                         ratio,
                                                         n_comp,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride);
                }
                if (ok && n_comp > 1u) {
                    const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
                    const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
                    const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
                    const uint64_t dst_plane_stride = 8ull * dst_row_stride;
                    ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         kvbuf,
                                                         ds4_gpu_tensor_offset(kv),
                                                         g_compressor_pool_kv_buffer,
                                                         dst_plane_stride,
                                                         head_dim,
                                                         ratio,
                                                         n_comp - 1u,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride) &&
                         ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         g_compressor_store_score_buffer,
                                                         0,
                                                         g_compressor_pool_score_buffer,
                                                         dst_plane_stride,
                                                         head_dim,
                                                         ratio,
                                                         n_comp - 1u,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride);
                }
                if (ok) {
                    ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                            comp_cache,
                                                            g_compressor_pool_kv_buffer,
                                                            0,
                                                            (uint64_t)head_dim * sizeof(float),
                                                            sizeof(float),
                                                            8ull * head_dim * sizeof(float),
                                                            g_compressor_pool_score_buffer,
                                                            0,
                                                            (uint64_t)head_dim * sizeof(float),
                                                            sizeof(float),
                                                            8ull * head_dim * sizeof(float),
                                                            8,
                                                            head_dim,
                                                            n_comp);
                }
            } else if (ok) {
                const uint64_t row_stride = (uint64_t)width * sizeof(float);
                ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                        comp_cache,
                                                        kvbuf,
                                                        ds4_gpu_tensor_offset(kv),
                                                        row_stride,
                                                        sizeof(float),
                                                        (uint64_t)ratio * row_stride,
                                                        g_compressor_store_score_buffer,
                                                        0,
                                                        row_stride,
                                                        sizeof(float),
                                                        (uint64_t)ratio * row_stride,
                                                        ratio,
                                                        head_dim,
                                                        n_comp);
            }
        }

        if (ok && n_comp != 0) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_comp != 0 && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && n_comp != 0 && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint32_t n_comp = n_tokens / ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor replay tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            ds4_gpu_tensor_bytes(comp_cache) < comp_bytes) {
            fprintf(stderr, "ds4: Metal compressor replay received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        const NSUInteger score_bytes = (NSUInteger)n_tokens * width * sizeof(float);
        const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
        if (ok && (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                    &g_compressor_store_score_bytes,
                                                    score_bytes,
                                                    "ds4_compressor_store_score") ||
                   !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                    &g_compressor_pool_kv_bytes,
                                                    pack_bytes,
                                                    "ds4_compressor_pool_kv") ||
                   !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                    &g_compressor_pool_score_bytes,
                                                    pack_bytes,
                                                    "ds4_compressor_pool_score"))) {
            ok = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                            scbuf,
                                                            ds4_gpu_tensor_offset(sc),
                                                            g_compressor_store_score_buffer,
                                                            0,
                                                            apebuf,
                                                            (NSUInteger)ape_inner,
                                                            ape_type,
                                                            width,
                                                            ratio,
                                                            pos0,
                                                            n_tokens);
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_kv_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_score_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                ds4_gpu_negative_infinity());
        }

        const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
        const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
        const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
        const uint64_t dst_plane_stride = 8ull * dst_row_stride;
        const NSUInteger state_off = ds4_gpu_tensor_offset(state_kv);
        const NSUInteger state_score_off = ds4_gpu_tensor_offset(state_score);

        if (ok) {
            /*
             * The aligned nonzero ratio-4 path replays the current ubatch
             * compressor, but seeds the first compressed row with the previous
             * compressor state. Rows 0..3 are the previous half, rows 4..7 are
             * the current half.
             */
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statekvbuf,
                                                 state_off,
                                                 g_compressor_pool_kv_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statescbuf,
                                                 state_score_off,
                                                 g_compressor_pool_score_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv) +
                                                         (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_kv_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_score_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok && n_comp > 1u) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv),
                                                 g_compressor_pool_kv_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 0,
                                                 g_compressor_pool_score_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok) {
            ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                    comp_cache,
                                                    g_compressor_pool_kv_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    g_compressor_pool_score_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    8,
                                                    head_dim,
                                                    n_comp);
        }
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                state_off,
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                state_score_off,
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const uint32_t prev_start = n_tokens - ratio;
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0 + prev_start,
                                                                 rows_prev,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill-state APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_tail);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_tail);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(sc_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor prefill-state received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows[4] = { 0, 1, 2, 3 };
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv_tail),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc_tail),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 rows,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_cur);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_cur);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        if (!kvbuf || !scbuf || !compbuf ||
            ds4_gpu_tensor_bytes(kv_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (emit && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor update received undersized buffers\n");
            return 0;
        }

        const bool use_store_one =
            getenv("DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE") == NULL;
        const int store_ok = use_store_one
            ? ds4_gpu_compressor_store_one_tensor(kv_cur,
                                                    sc_cur,
                                                    state_kv,
                                                    state_score,
                                                    model_map,
                                                    model_size,
                                                    ape_offset,
                                                    ape_type,
                                                    width,
                                                    ratio,
                                                    pos)
            : ds4_gpu_compressor_store_batch_tensor(kv_cur,
                                                      sc_cur,
                                                      state_kv,
                                                      state_score,
                                                      model_map,
                                                      model_size,
                                                      ape_offset,
                                                      ape_type,
                                                      head_dim,
                                                      ratio,
                                                      pos,
                                                      1);
        if (!store_ok) {
            return 0;
        }
        if (!emit) return 1;

        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                comp_cache,
                (uint64_t)comp_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
        if (!comp_row_view) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_compressor_pool(cb,
                                                  comp_row_view,
                                                  state_kv,
                                                  state_score,
                                                  head_dim,
                                                  ratio);
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor DS4 softmax pool");
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view,
                                                       comp_row_view,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       1,
                                                       rms_eps) != 0;
        }
        if (ok) {
            const uint32_t comp_pos = pos + 1u - ratio;
            ok = ds4_gpu_rope_tail_tensor(comp_row_view,
                                            1,
                                            1,
                                            head_dim,
                                            n_rot,
                                            comp_pos,
                                            n_ctx_orig,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            beta_fast,
                                            beta_slow) != 0;
        }
        if (ok && ratio == 4u) {
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb &&
                 ds4_gpu_encode_compressor_shift_ratio4(cb,
                                                          state_kv,
                                                          state_score,
                                                          width);
            if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor ratio4 state shift");
        }
        ds4_gpu_tensor_free(comp_row_view);
        if (!ok) return 0;
    }

    return 1;
}

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value) {
    if (!cb || !buf || width == 0 || rows == 0 || (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, 1, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth_max = g_unary_fill_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !group_tmp || !low_tmp || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || (low_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output batch received invalid q8 dimensions\n");
            return 0;
        }
        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t row_b_bytes = (low_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        const uint64_t out_b_bytes = out_dim * row_b_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
            out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) {
            fprintf(stderr, "ds4: Metal attention output batch weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_tokens * n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_tokens * low_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal attention output batch received undersized buffers\n");
            return 0;
        }
        (void)group_tmp;
        (void)low_tmp;

        const bool use_direct_low =
            n_tokens < 32u && getenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == NULL;
        /* The exported TensorOps attention-output kernel is a 64-token tile.
         * Keep this on full tiles only; smaller multiples of 32 use the legacy
         * path instead of relying on cooperative tensor partial RHS bounds. */
        const bool use_mpp_low =
            n_tokens >= 32u &&
            (n_tokens % DS4_METAL_ATTN_OUT_MPP_TILE_N) == 0 &&
            ds4_gpu_use_mpp_attn_out_low_matmul();
        const NSUInteger ids_bytes = (NSUInteger)n_tokens * (NSUInteger)n_groups * sizeof(int32_t);
        id<MTLBuffer> group_ids_buffer = nil;
        if (!use_direct_low && !use_mpp_low) {
            if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                group_ids_buffer =
                    ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                if (!group_ids_buffer) {
                    return 0;
                }
            } else {
                if (!ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                     &g_attn_out_group_ids_bytes,
                                                     ids_bytes,
                                                     "ds4_attention_output_group_ids")) {
                    return 0;
                }
                group_ids_buffer = g_attn_out_group_ids_buffer;
            }
            int32_t *ids = (int32_t *)[group_ids_buffer contents];
            for (uint32_t t = 0; t < n_tokens; t++) {
                for (uint32_t group = 0; group < n_groups; group++) {
                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                }
            }
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }
        const bool attn_out_profile =
            getenv("DS4_METAL_ATTN_OUT_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        if (ok && attn_out_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                ok = false;
            } else {
                cb = ds4_gpu_command_buffer(&owned);
                if (!cb || owned) ok = false;
            }
        }
        double attn_out_t0 = attn_out_profile ? ds4_gpu_now_ms() : 0.0;
#define DS4_METAL_PROFILE_ATTN_OUT_STAGE(name) do { \
            if (ok && attn_out_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = false; \
                } else { \
                    const double now_ms = ds4_gpu_now_ms(); \
                    fprintf(stderr, \
                            "ds4: Metal attention output stage tokens=%u %s=%.3f ms\n", \
                            n_tokens, (name), now_ms - attn_out_t0); \
                    attn_out_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = false; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb || owned) ok = false; \
                    } \
                } \
            } \
        } while (0)

        if (ok) {
            /*
             * Batched attention-output projections switch from the vector
             * kernel to the SIMD matrix kernel once the batch has at least 32
             * tokens.  This preserves the single-token generation path while
             * keeping prefill accumulation stable.
             */
            if (use_mpp_low) {
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                /*
                 * Direct RHS lets MPP read the dense low-rank activation tile
                 * directly from device memory instead of staging a second
                 * threadgroup tile.  The retained attention-output path is the
                 * 64-token direct-RHS kernel; the older staged-RHS and 32-token
                 * variants were not kept as alternate runtime modes.
                 */
                const char *attn_out_pipeline_name =
                    "kernel_attn_out_low_q8_0_mpp_direct_rhs_n64";
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline(attn_out_pipeline_name, false);
                ok = ds4_gpu_encode_attn_out_low_q8_mpp(cb,
                                                          mm_pipeline,
                                                          &mm_args,
                                                          out_a_buf,
                                                          (NSUInteger)out_a_inner,
                                                          ds4_gpu_tensor_buffer(heads),
                                                          ds4_gpu_tensor_offset(heads),
                                                          ds4_gpu_tensor_buffer(low),
                                                          ds4_gpu_tensor_offset(low)) != 0;
                if (!ok) {
                    ds4_gpu_warn_mpp_fallback();
                    if (ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                        if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                            group_ids_buffer =
                                ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                        } else if (ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                                   &g_attn_out_group_ids_bytes,
                                                                   ids_bytes,
                                                                   "ds4_attention_output_group_ids")) {
                            group_ids_buffer = g_attn_out_group_ids_buffer;
                        }
                        if (group_ids_buffer) {
                            int32_t *ids = (int32_t *)[group_ids_buffer contents];
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                for (uint32_t group = 0; group < n_groups; group++) {
                                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                                }
                            }
                            ds4_gpu_mul_mm_id_map_args map_args =
                                ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                                  n_groups,
                                                                  n_groups,
                                                                  n_groups,
                                                                  n_tokens);
                            id<MTLComputePipelineState> map_pipeline =
                                ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                            id<MTLComputePipelineState> fallback_pipeline =
                                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                            ok = ds4_gpu_encode_mul_mm_id(cb,
                                                            map_pipeline,
                                                            fallback_pipeline,
                                                            &map_args,
                                                            &mm_args,
                                                            out_a_buf,
                                                            (NSUInteger)out_a_inner,
                                                            ds4_gpu_tensor_buffer(heads),
                                                            ds4_gpu_tensor_offset(heads),
                                                            ds4_gpu_tensor_buffer(low),
                                                            ds4_gpu_tensor_offset(low),
                                                            group_ids_buffer,
                                                            0) != 0;
                        }
                    }
                }
            } else if (n_tokens >= 32u && ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                ds4_gpu_mul_mm_id_map_args map_args =
                    ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                      n_groups,
                                                      n_groups,
                                                      n_groups,
                                                      n_tokens);
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                id<MTLComputePipelineState> map_pipeline =
                    ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                ok = ds4_gpu_encode_mul_mm_id(cb,
                                                map_pipeline,
                                                mm_pipeline,
                                                &map_args,
                                                &mm_args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0) != 0;
            } else if (use_direct_low) {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = 0,
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
                ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                             pipeline,
                                                             &args,
                                                             out_a_buf,
                                                             (NSUInteger)out_a_inner,
                                                             ds4_gpu_tensor_buffer(heads),
                                                             ds4_gpu_tensor_offset(heads),
                                                             ds4_gpu_tensor_buffer(low),
                                                             ds4_gpu_tensor_offset(low),
                                                             32u * 2u * sizeof(float),
                                                             4) != 0;
            } else {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = (uint64_t)n_groups * sizeof(int32_t),
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_0_f32", 4);
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                pipeline,
                                                &args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0,
                                                32u * 2u * sizeof(float),
                                                4,
                                                true) != 0;
            }
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("low_proj");

        if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                              out_b_offset,
                                              low_dim, out_dim, low, n_tokens) != 0;
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("out_proj");

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
#undef DS4_METAL_PROFILE_ATTN_OUT_STAGE
        return ok ? 1 : 0;
    }
}

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output low received invalid q8 dimensions\n");
            return 0;
        }

        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) {
            fprintf(stderr, "ds4: Metal attention output low weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = low_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes) {
            fprintf(stderr, "ds4: Metal attention output low received undersized buffers\n");
            return 0;
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }

        if (ok) {
            ds4_gpu_mul_mv_id_args args = {
                .nei0 = (int32_t)n_groups,
                .nei1 = 1,
                .nbi1 = 0,
                .ne00 = (int32_t)group_dim,
                .ne01 = (int32_t)rank,
                .ne02 = (int32_t)n_groups,
                .nb00 = 34,
                .nb01 = row_a_bytes,
                .nb02 = (uint64_t)rank * row_a_bytes,
                .ne10 = (int32_t)group_dim,
                .ne11 = (int32_t)n_groups,
                .ne12 = 1,
                .ne13 = 1,
                .nb10 = sizeof(float),
                .nb11 = (uint64_t)group_dim * sizeof(float),
                .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                .ne0 = (int32_t)rank,
                .ne1 = (int32_t)n_groups,
                .nb1 = (uint64_t)rank * sizeof(float),
                .nr0 = 2,
            };
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
            ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                         pipeline,
                                                         &args,
                                                         out_a_buf,
                                                         (NSUInteger)out_a_inner,
                                                         ds4_gpu_tensor_buffer(heads),
                                                         ds4_gpu_tensor_offset(heads),
                                                         ds4_gpu_tensor_buffer(low),
                                                         ds4_gpu_tensor_offset(low),
                                                         32u * 2u * sizeof(float),
                                                         4) != 0;
        }

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        return ok ? 1 : 0;
    }
}

static NSUInteger ds4_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = src_col_stride,
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f16_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = (uint64_t)rows * src_row_stride,
        .nb03 = (uint64_t)rows * src_row_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = (uint64_t)rows * dst_row_stride,
        .nb3 = (uint64_t)rows * dst_row_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f16_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_copy_to_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        bool                 src_is_f16,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst) return 0;
    if (n == 0) return 1;
    if (!src_is_f16) {
        return ds4_gpu_encode_cpy_f32_f16_1d(cb, src, src_off, dst, dst_off, n);
    }

    if (g_batch_cb && cb == g_batch_cb) ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:src
            sourceOffset:src_off
                toBuffer:dst
       destinationOffset:dst_off
                    size:(NSUInteger)n * sizeof(uint16_t)];
    [blit endEncoding];
    return 1;
}

static int ds4_gpu_encode_fill_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             n,
        float                value) {
    if (!cb || !buf || n == 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(n, 1, 0, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth = (NSUInteger)n;
    const NSUInteger max_threads = g_unary_fill_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_raw_heads(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *heads,
        id<MTLBuffer>         sinks_buf,
        NSUInteger            sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || raw_cap < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t heads_bytes = q_bytes;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < heads_bytes) {
        fprintf(stderr, "ds4: Metal DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_raw, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }
    memset([mask_buffer contents], 0, mask_bytes);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_raw % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, (n_raw % ncpsg) != 0,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        const uint32_t tail_elems = tail_rows * head_dim;
        const uint32_t head_elems = head_rows * head_dim;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }

        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_elems)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_elems))) {
            return 0;
        }

        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }

    if ((n_raw % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static void ds4_gpu_fill_raw_prefill_mask(uint16_t *mask, uint32_t n_tokens, uint32_t window) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_tokens;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= q;
            const bool in_window = window == 0 || q - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_raw_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  pos0,
        uint32_t  window) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* The caller has already copied the SWA ring into logical order when it
     * wraps, so key row k represents first_raw_pos + k. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_raw;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_mixed_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  n_comp,
        uint32_t  pos0,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* Raw keys are laid out by logical position; compressed keys follow them. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
        const uint32_t n_visible = (qpos + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_raw + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_static_mixed_prefill_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_comp,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_tokens + n_comp;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= q;
            const bool in_window = window == 0 || q - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }

        const uint32_t n_visible = (q + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_tokens + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer = ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "static_mixed_nonvec", (name), n_tokens, n_comp, n_keys, \
                    n_head, head_dim, window, ratio, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");
    if (n_comp &&
        !ds4_gpu_encode_copy_to_f16_1d(cb,
                                       compbuf,
                                       ds4_gpu_tensor_offset(comp_kv),
                                       comp_kv_f16 != 0,
                                       g_flash_attn_kv_buffer,
                                       (NSUInteger)n_tokens * row_bytes_f16,
                                       n_comp * head_dim)) {
        return 0;
    }
    if (n_comp) {
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_comp");
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_comp_copy");
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = nil;
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("block_map");

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "static_mixed_vec", (name), n_tokens, n_comp, n_keys, \
                    n_head, head_dim, window, ratio, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");
    if (n_comp) {
        if (!ds4_gpu_encode_copy_to_f16_1d(cb,
                                           compbuf,
                                           ds4_gpu_tensor_offset(comp_kv),
                                           comp_kv_f16 != 0,
                                           g_flash_attn_kv_buffer,
                                           (NSUInteger)n_tokens * row_bytes_f16,
                                           n_comp * head_dim)) {
            return 0;
        }
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_comp");
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_comp_copy");
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, has_kvpad,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_vec");

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_reduce");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (n_tokens >= 20) {
        return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(cbp,
                                                                                       heads,
                                                                                       sinks_buf,
                                                                                       sinks_offset,
                                                                                       q,
                                                                                       raw_kv,
                                                                                       comp_kv,
                                                                                       comp_kv_f16,
                                                                                       comp_mask,
                                                                                       use_comp_mask,
                                                                                       n_tokens,
                                                                                       n_comp,
                                                                                       window,
                                                                                       ratio,
                                                                                       n_head,
                                                                                       head_dim);
    }
    return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(cbp,
                                                                           heads,
                                                                           sinks_buf,
                                                                           sinks_offset,
                                                                           q,
                                                                           raw_kv,
                                                                           comp_kv,
                                                                           comp_kv_f16,
                                                                           comp_mask,
                                                                           use_comp_mask,
                                                                           n_tokens,
                                                                           n_comp,
                                                                           window,
                                                                           ratio,
                                                                           n_head,
                                                                           head_dim);
}

static int ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_tokens == 0) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill raw DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_tokens % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_tokens * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_tokens + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "raw_nonvec", (name), n_tokens, 0, n_tokens, \
                    n_head, head_dim, window, 0, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents], n_tokens, window);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_tokens,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_tokens * row_bytes_f16,
            .nb13 = (uint64_t)n_tokens * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_tokens * row_bytes_f16,
            .nb23 = (uint64_t)n_tokens * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_tokens,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("block_map");

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_tokens,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_tokens * row_bytes_f16,
        .nb13 = (uint64_t)n_tokens * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_tokens * row_bytes_f16,
        .nb23 = (uint64_t)n_tokens * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_raw_heads(
        id<MTLCommandBuffer> __strong *cbp,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (!cbp || !*cbp) return 0;
    id<MTLCommandBuffer> cb = *cbp;
    if (head_dim != 512 || n_head == 0 || n_tokens == 0) {
        return 0;
    }
    if (n_tokens >= 20) {
        return ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec(cbp,
                                                                         heads,
                                                                         sinks_buf,
                                                                         sinks_offset,
                                                                         q,
                                                                         raw_kv,
                                                                         n_tokens,
                                                                         window,
                                                                         n_head,
                                                                         head_dim);
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill raw DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_tokens, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_f16_offset = 0;
    const NSUInteger kv_f16_bytes = (NSUInteger)n_tokens * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_f16_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    const bool flash_stage_profile =
        getenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE") != NULL && g_batch_cb != nil;
    double flash_stage_t0 = 0.0;
    if (flash_stage_profile) {
        if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
            return 0;
        }
        int profile_owned = 0;
        cb = ds4_gpu_command_buffer(&profile_owned);
        if (!cb || profile_owned) return 0;
        *cbp = cb;
        flash_stage_t0 = ds4_gpu_now_ms();
    }
#define DS4_METAL_PROFILE_FLASH_ATTN_STAGE(name) do { \
        if (flash_stage_profile) { \
            if (!ds4_gpu_flash_attn_stage_profile_boundary(cbp, \
                    "raw_vec", (name), n_tokens, 0, n_tokens, \
                    n_head, head_dim, window, 0, &flash_stage_t0)) { \
                return 0; \
            } \
            cb = *cbp; \
        } \
    } while (0)

    ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents], n_tokens, window);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("mask_fill");

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_tokens % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, true,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         kv_f16_offset,
                                         n_tokens * head_dim)) {
        return 0;
    }
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("copy_raw");

    if ((n_tokens % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_tokens,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_tokens * row_bytes_f16,
            .nb13 = (uint64_t)n_tokens * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_tokens * row_bytes_f16,
            .nb23 = (uint64_t)n_tokens * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        DS4_METAL_PROFILE_FLASH_ATTN_STAGE("pad");
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_tokens,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_tokens * row_bytes_f16,
        .nb13 = (uint64_t)n_tokens * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_tokens * row_bytes_f16,
        .nb23 = (uint64_t)n_tokens * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_vec");

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    DS4_METAL_PROFILE_FLASH_ATTN_STAGE("attention_reduce");

#undef DS4_METAL_PROFILE_FLASH_ATTN_STAGE
    return 1;
}

static int ds4_gpu_encode_flash_attention_gathered_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        uint32_t               n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_mask,
        uint32_t               n_head,
        uint32_t               head_dim) {
    const uint32_t n_keys = n_raw + n_comp;
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || n_keys == 0 ||
        raw_cap < n_raw || n_keys < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : nil;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : nil;
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        (n_comp && !compbuf) ||
        (use_mask && !maskbuf) ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes ||
        (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes)) {
        fprintf(stderr, "ds4: Metal gathered DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_mask_buffer,
                                         &g_flash_attn_mask_bytes,
                                         mask_bytes,
                                         "ds4_flash_attn_mask") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_keys % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, (n_keys % ncpsg) != 0,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLBuffer> raw_linear_buf = rawbuf;
    NSUInteger raw_linear_offset = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_rows = raw_cap - raw_start < n_raw ? raw_cap - raw_start : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }

        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_rows * head_dim)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_rows * head_dim))) {
            return 0;
        }

        raw_linear_buf = g_flash_attn_ring_buffer;
        raw_linear_offset = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         raw_linear_buf,
                                         raw_linear_offset,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }
    if (n_comp) {
        if (!ds4_gpu_encode_copy_to_f16_1d(cb,
                                           compbuf,
                                           ds4_gpu_tensor_offset(comp_kv),
                                           comp_kv_f16 != 0,
                                           g_flash_attn_kv_buffer,
                                           (NSUInteger)n_raw * row_bytes_f16,
                                           n_comp * head_dim)) {
            return 0;
        }
    }

    if (!ds4_gpu_encode_fill_f16_1d(cb, g_flash_attn_mask_buffer, 0, n_keys, 0.0f)) {
        return 0;
    }
    if (use_mask && n_comp &&
        !ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         maskbuf,
                                         ds4_gpu_tensor_offset(comp_mask),
                                         g_flash_attn_mask_buffer,
                                         (NSUInteger)n_raw * sizeof(uint16_t),
                                         n_comp)) {
        return 0;
    }

    if ((n_keys % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:g_flash_attn_mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:g_flash_attn_mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_decode_raw_batch_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal decode raw batch FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_raw % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_raw + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }
        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_rows * head_dim)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_rows * head_dim))) {
            return 0;
        }
        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }

    ds4_gpu_fill_raw_decode_batch_mask((uint16_t *)[mask_buffer contents],
                                         n_tokens,
                                         n_raw,
                                         pos0,
                                         window);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_raw,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (n_comp == 0) {
        return ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       sinks_offset,
                                                                       q,
                                                                       raw_kv,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       window,
                                                                       n_head,
                                                                       head_dim);
    }
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || !comp_kv || (use_comp_mask && !comp_mask)) {
        return 0;
    }

    const uint32_t n_keys = n_raw + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal decode mixed batch FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }
        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_rows * head_dim)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_rows * head_dim))) {
            return 0;
        }
        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim) ||
        !ds4_gpu_encode_copy_to_f16_1d(cb,
                                       compbuf,
                                       ds4_gpu_tensor_offset(comp_kv),
                                       comp_kv_f16 != 0,
                                       g_flash_attn_kv_buffer,
                                       (NSUInteger)n_raw * row_bytes_f16,
                                       n_comp * head_dim)) {
        return 0;
    }

    ds4_gpu_fill_mixed_decode_batch_mask((uint16_t *)[mask_buffer contents],
                                           n_tokens,
                                           n_raw,
                                           n_comp,
                                           pos0,
                                           window,
                                           ratio);
    if (use_comp_mask) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_raw * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0) return 0;

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_raw_heads(&cb,
                                                                heads,
                                                                sinks_buf,
                                                                (NSUInteger)sinks_inner,
                                                                q,
                                                                raw_kv,
                                                                n_tokens,
                                                                window,
                                                                n_head,
                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill raw attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                     heads,
                                                                     sinks_buf,
                                                                     (NSUInteger)sinks_inner,
                                                                     q,
                                                                     raw_kv,
                                                                     n_tokens,
                                                                     pos0,
                                                                     n_raw,
                                                                     raw_cap,
                                                                     raw_start,
                                                                     window,
                                                                     n_head,
                                                                     head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode raw batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || (n_comp != 0 && !comp_kv) ||
        (use_comp_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       (NSUInteger)sinks_inner,
                                                                       q,
                                                                       raw_kv,
                                                                       comp_kv,
                                                                       comp_kv_f16,
                                                                       comp_mask,
                                                                       use_comp_mask,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       n_comp,
                                                                       window,
                                                                       ratio,
                                                                       n_head,
                                                                       head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode mixed batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_kv || !topk ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 || top_k > n_comp || (top_k & (top_k - 1u)) != 0 ||
        ratio == 0 || n_head == 0 || head_dim != 512) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal indexed attention sinks range is outside the mapped model\n");
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t row_bytes_f16 = (uint64_t)head_dim * sizeof(uint16_t);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * row_bytes;
        const uint64_t raw_bytes = (uint64_t)raw_cap * row_bytes;
        const uint64_t comp_bytes = (uint64_t)n_comp * (comp_kv_f16 ? row_bytes_f16 : row_bytes);
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        if (!qbuf || !rawbuf || !compbuf || !topkbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal indexed mixed attention received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        id<MTLComputePipelineState> sort_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline,
                                    "kernel_dsv4_sort_i32_rows_asc");
        const bool decode_one_token = n_tokens == 1u;
        id<MTLComputePipelineState> attn_pipeline =
            decode_one_token ?
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_rb16_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8_rb16") :
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8");
        if (!sort_pipeline || !attn_pipeline) return 0;
        if ((NSUInteger)top_k > sort_pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal indexed attention top-k exceeds sort threadgroup limit\n");
            return 0;
        }
        /*
         * Fast decode attends to the same full top-k compressed rows but keeps
         * them in score order, avoiding a chronological sort dispatch.
         * --quality restores the sorted order for stricter reproducibility.
         */
        const bool skip_decode_sort = !g_quality_mode && decode_one_token;
        if (!skip_decode_sort &&
            !ds4_gpu_ensure_scratch_buffer(&g_indexed_topk_buffer,
                                             &g_indexed_topk_bytes,
                                             (NSUInteger)topk_bytes,
                                             "ds4_indexed_topk_sorted")) {
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args sort_args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)top_k,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(int32_t),
            .nb1 = (uint64_t)top_k * sizeof(int32_t),
        };
        ds4_gpu_dsv4_indexed_attention_args attn_args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .n_raw = n_raw,
            .raw_cap = raw_cap,
            .raw_start = raw_start,
            .n_comp = n_comp,
            .top_k = top_k,
            .pos0 = pos0,
            .window = window,
            .ratio = ratio,
            .comp_kv_f16 = comp_kv_f16 ? 1u : 0u,
            .pad0 = 0,
            .q_token_stride = (uint64_t)n_head * row_bytes,
            .q_head_stride = row_bytes,
            .raw_row_stride = row_bytes,
            .comp_row_stride = comp_kv_f16 ? row_bytes_f16 : row_bytes,
            .topk_token_stride = (uint64_t)top_k * sizeof(int32_t),
            .dst_token_stride = (uint64_t)n_head * row_bytes,
            .dst_head_stride = row_bytes,
            .scale = 1.0f / sqrtf((float)head_dim),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = nil;
        if (!skip_decode_sort) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:sort_pipeline];
            [enc setBytes:&sort_args length:sizeof(sort_args) atIndex:0];
            [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
            [enc setBuffer:g_indexed_topk_buffer offset:0 atIndex:2];
            [enc setThreadgroupMemoryLength:(NSUInteger)top_k * sizeof(int32_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(top_k, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_kv) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(comp_kv) atIndex:3];
        [enc setBuffer:skip_decode_sort ? topkbuf : g_indexed_topk_buffer
              offset:skip_decode_sort ? ds4_gpu_tensor_offset(topk) : 0
             atIndex:4];
        [enc setBuffer:sinks_buf offset:(NSUInteger)sinks_inner atIndex:5];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:6];
        [enc setThreadgroupMemoryLength:(decode_one_token ? 16u : 1u) *
                                        128u * 4u * sizeof(uint16_t)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, ((NSUInteger)n_head + 7u) / 8u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph indexed mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        ratio == 0 || (n_comp != 0 && !comp_kv)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(&cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_kv_f16,
                                                                                NULL,
                                                                                0,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill static mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !comp_kv || !comp_mask || !model_map ||
        n_tokens == 0 || n_comp == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(&cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_kv_f16,
                                                                                comp_mask,
                                                                                1,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill masked mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        n_raw > UINT32_MAX - n_comp || n_raw + n_comp > 8192u ||
        (n_comp != 0 && !comp_kv) ||
        (use_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim *
                                    (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float));
        const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
        if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal graph attention heads sink range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
        id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
        if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
            (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal graph attention heads received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size, sinks_offset, sink_bytes, &sinks_inner);
        if (!sinks_buf) return 0;

        if (n_comp == 0) {
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;

            if (!ds4_gpu_encode_flash_attention_raw_heads(cb,
                                                            heads,
                                                            sinks_buf,
                                                            (NSUInteger)sinks_inner,
                                                            q,
                                                            raw_kv,
                                                            n_raw,
                                                            raw_cap,
                                                            raw_start,
                                                            n_head,
                                                            head_dim)) {
                return 0;
            }

            if (!ds4_gpu_finish_command_buffer(cb, owned, "graph raw attention heads")) return 0;
            return 1;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_gathered_heads(cb,
                                                             heads,
                                                             sinks_buf,
                                                             (NSUInteger)sinks_inner,
                                                             q,
                                                             raw_kv,
                                                             n_raw,
                                                             raw_cap,
                                                             raw_start,
                                                             comp_kv,
                                                             comp_kv_f16,
                                                             n_comp,
                                                             comp_mask,
                                                             use_mask,
                                                             n_head,
                                                             head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (!isfinite(clamp) || clamp < 0.0f || !isfinite(weight)) return 0;

    @autoreleasepool {
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!gatebuf || !upbuf || !outbuf ||
            ds4_gpu_tensor_bytes(gate) < bytes ||
            ds4_gpu_tensor_bytes(up) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal SwiGLU received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glu_args args = {
            .ne00 = (int32_t)n,
            .nb01 = (uint64_t)n * sizeof(float),
            .ne10 = (int32_t)n,
            .nb11 = (uint64_t)n * sizeof(float),
            .ne0 = (int32_t)n,
            .nb1 = (uint64_t)n * sizeof(float),
            .i00 = 0,
            .i10 = 0,
            .alpha = weight,
            .limit = clamp,
        };
        NSUInteger nth = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger ds4_nth = n > 1 ? (NSUInteger)n / 2u : 1u;
        if (nth > ds4_nth) nth = ds4_nth;
        if (nth == 0) nth = 1;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "SwiGLU")) return 0;
    }

    return 1;
}

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = ds4_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !outbuf ||
            ds4_gpu_tensor_bytes(a) < bytes ||
            ds4_gpu_tensor_bytes(b) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal tensor add received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        ds4_gpu_bin_args args = {
            .ne00 = (int32_t)n,
            .ne01 = 1,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = row_bytes,
            .nb02 = row_bytes,
            .nb03 = row_bytes,
            .ne10 = (int32_t)n,
            .ne11 = 1,
            .ne12 = 1,
            .ne13 = 1,
            .nb10 = sizeof(float),
            .nb11 = row_bytes,
            .nb12 = row_bytes,
            .nb13 = row_bytes,
            .ne0 = (int32_t)n,
            .ne1 = 1,
            .ne2 = 1,
            .ne3 = 1,
            .nb0 = sizeof(float),
            .nb1 = row_bytes,
            .nb2 = row_bytes,
            .nb3 = row_bytes,
            .offs = 0,
            .o1 = { 0 },
        };
        NSUInteger nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth_max > 256u) nth_max = 256u;
        NSUInteger nth = 1;
        while (2u * nth < (NSUInteger)args.ne0 && nth < nth_max) {
            nth *= 2u;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:ds4_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:ds4_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "tensor add")) return 0;
    }

    return 1;
}

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
} ds4_gpu_directional_steering_project_args;

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_directional_steering_project_f32");
        if (!pipeline) return 0;

        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> dbuf = ds4_gpu_tensor_buffer(directions);
        const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
        const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
        if (!xbuf || !dbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(directions) < dir_bytes) {
            fprintf(stderr, "ds4: Metal directional steering received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        while (nth > width && nth > 1u) nth >>= 1;
        if (nth == 0) nth = 1;

        ds4_gpu_directional_steering_project_args args = {
            .width = width,
            .rows = rows,
            .layer = layer,
            .n_threads = (uint32_t)nth,
            .scale = scale,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:dbuf offset:ds4_gpu_tensor_offset(directions) atIndex:2];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "directional steering")) return 0;
    }

    return 1;
}

static NSUInteger ds4_gpu_bin_threads(uint32_t width, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)width && nth < nth_max) nth *= 2u;
    return nth ? nth : 1u;
}

static int ds4_gpu_encode_unary_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer>               src,
        NSUInteger                  src_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        uint32_t                    width,
        uint32_t                    rows,
        int                         c4,
        float                       min,
        float                       max) {
    if (!cb || !pipeline || !src || !dst || width == 0 || rows == 0) return 0;
    if (c4 && (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, c4, 0.0f, 0.0f);
    args.min = min;
    args.max = max;

    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off) {
    if (!cb || !pipeline || !args || !a || !b || !out || args->ne0 <= 0 || args->ne1 <= 0) {
        return 0;
    }

    const NSUInteger nth = ds4_gpu_bin_threads((uint32_t)args->ne0, pipeline);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)args->ne1,
                                          (NSUInteger)args->ne2,
                                          (NSUInteger)args->ne3)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_rowwise_scalar_args(uint32_t width, uint32_t rows) {
    const uint64_t lhs_row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rhs_row_bytes = sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = lhs_row_bytes,
        .nb02 = (uint64_t)rows * lhs_row_bytes,
        .nb03 = (uint64_t)rows * lhs_row_bytes,
        .ne10 = 1,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = (uint64_t)rows * rhs_row_bytes,
        .nb13 = (uint64_t)rows * rhs_row_bytes,
        .ne0 = (int32_t)width,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = lhs_row_bytes,
        .nb2 = (uint64_t)rows * lhs_row_bytes,
        .nb3 = (uint64_t)rows * lhs_row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_mul_mv_id_args ds4_gpu_make_mul_mv_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t nr0) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    const uint64_t src0_blocks = src0_cols / 256u;
    const uint64_t src0_block_bytes = src0_blocks ? src0_row_bytes / src0_blocks : 1u;
    return (ds4_gpu_mul_mv_id_args) {
        .nei0 = (int32_t)selected_experts,
        .nei1 = (int32_t)n_tokens,
        .nbi1 = (uint64_t)selected_experts * sizeof(int32_t),
        .ne00 = (int32_t)src0_cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = (int32_t)src0_experts,
        .nb00 = src0_block_bytes,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .ne12 = (int32_t)n_tokens,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .nb1 = (uint64_t)src0_rows * sizeof(float),
        .nr0 = (int32_t)nr0,
    };
}

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    return (ds4_gpu_mul_mm_id_map_args) {
        .ne02 = (int32_t)src0_experts,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne21 = (int32_t)n_tokens,
        .ne20 = (int32_t)selected_experts,
        .nb21 = (uint64_t)selected_experts * sizeof(int32_t),
    };
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    return ds4_gpu_make_mul_mm_id_args_src1_size(src0_cols,
                                                   src0_rows,
                                                   src0_experts,
                                                   src0_row_bytes,
                                                   src0_expert_bytes,
                                                   src1_expert_rows,
                                                   selected_experts,
                                                   n_tokens,
                                                   sizeof(float));
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * src1_elem_size;
    return (ds4_gpu_mul_mm_id_args) {
        .ne00 = (int32_t)src0_cols,
        .ne02 = (int32_t)src0_experts,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .nb03 = (uint64_t)src0_experts * src0_expert_bytes,
        .ne11 = (int32_t)src1_expert_rows,
        .nb10 = src1_elem_size,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .nb13 = (uint64_t)n_tokens * (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne20 = (int32_t)selected_experts,
        .ne21 = (int32_t)n_tokens,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .r2 = 1,
        .r3 = 1,
    };
}

static uint32_t ds4_gpu_routed_mv_nr0(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q4_K:    return 2;
    case DS4_METAL_TENSOR_Q2_K:
    case DS4_METAL_TENSOR_IQ2_XXS: return 4;
    default:                       return 0;
    }
}

static const char *ds4_gpu_metal_tensor_type_name(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return "iq2_xxs";
    case DS4_METAL_TENSOR_Q2_K:    return "q2_k";
    case DS4_METAL_TENSOR_Q4_K:    return "q4_k";
    default:                       return "unknown";
    }
}

static NSUInteger ds4_gpu_routed_mv_smem(uint32_t type) {
    if (type == DS4_METAL_TENSOR_IQ2_XXS) {
        return 256u * sizeof(uint64_t) + 128u * sizeof(uint8_t);
    }
    return 0;
}

static id<MTLComputePipelineState> ds4_gpu_routed_mv_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return g_moe_mul_mv_id_iq2_xxs_pipeline;
    case DS4_METAL_TENSOR_Q2_K:    return g_moe_mul_mv_id_q2_k_pipeline;
    case DS4_METAL_TENSOR_Q4_K:    return g_moe_mul_mv_id_q4_k_pipeline;
    default:                       return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f32", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f32", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f16", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f16", false);
    default:
        return nil;
    }
}

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_pair(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0_a offset:src0_a_off atIndex:1];
    [enc setBuffer:src0_b offset:src0_b_off atIndex:2];
    [enc setBuffer:src1   offset:src1_off   atIndex:3];
    [enc setBuffer:dst_a  offset:dst_a_off  atIndex:4];
    [enc setBuffer:dst_b  offset:dst_b_off  atIndex:5];
    [enc setBuffer:ids    offset:ids_off    atIndex:6];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act ||
        !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:src0_a  offset:src0_a_off  atIndex:2];
    [enc setBuffer:src0_b  offset:src0_b_off  atIndex:3];
    [enc setBuffer:src1    offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids     offset:ids_off     atIndex:8];
    [enc setBuffer:weights offset:weights_off atIndex:9];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !mm_pipeline || !map_args || !mm_args ||
        !src0 || !src1 || !dst || !ids ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    return ds4_gpu_encode_mul_mm_id_map(cb,
                                          map_pipeline,
                                          map_args,
                                          mm_args,
                                          ids,
                                          ids_off) &&
           ds4_gpu_encode_mul_mm_id_mapped(cb,
                                             mm_pipeline,
                                             mm_args,
                                             src0,
                                             src0_off,
                                             src1,
                                             src1_off,
                                             dst,
                                             dst_off);
}

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !map_args || !mm_args || !ids ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) return 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer,
                                         &g_moe_id_map_bytes,
                                         tpe_bytes + hids_bytes,
                                         "ds4_moe_id_map")) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:map_pipeline];
    [enc setBytes:map_args length:sizeof(*map_args) atIndex:0];
    [enc setBuffer:ids offset:ids_off atIndex:1];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:(NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne20 * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake((NSUInteger)mm_args->ne02, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    if (!cb || !mm_pipeline || !mm_args || !src0 || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }
    /*
     * The routed MoE grouped matmul uses the legacy 32-token expert-major tile.
     * The removed TensorOps variant was not semantically stable on evals, so keep
     * this encoder tied to the tested simdgroup kernel shape.
     */
    const NSUInteger tile_n = 32u;

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc setThreadgroupMemoryLength:8192u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + tile_n - 1u) / tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    return ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                  mm_pipeline,
                                                  mm_args,
                                                  src0,
                                                  src0_off,
                                                  src1,
                                                  src1_off,
                                                  dst,
                                                  dst_off);
}

static int ds4_gpu_encode_attn_out_low_q8_mpp(
        id<MTLCommandBuffer>           cb,
        id<MTLComputePipelineState>    pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>                  src0,
        NSUInteger                     src0_off,
        id<MTLBuffer>                  src1,
        NSUInteger                     src1_off,
        id<MTLBuffer>                  dst,
        NSUInteger                     dst_off) {
    if (!cb || !pipeline || !mm_args || !src0 || !src1 || !dst ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne02 <= 0 || mm_args->ne1 <= 0 || mm_args->ne21 <= 0) {
        return 0;
    }

    const uint32_t tile_n = DS4_METAL_ATTN_OUT_MPP_TILE_N;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst offset:dst_off atIndex:3];
    [enc setThreadgroupMemoryLength:4096u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + (NSUInteger)tile_n - 1u) / (NSUInteger)tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_swiglu_flat(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !gate || !up || !out || n == 0) return 0;

    ds4_gpu_glu_args args = {
        .ne00 = (int32_t)n,
        .nb01 = (uint64_t)n * sizeof(float),
        .ne10 = (int32_t)n,
        .nb11 = (uint64_t)n * sizeof(float),
        .ne0 = (int32_t)n,
        .nb1 = (uint64_t)n * sizeof(float),
        .i00 = 0,
        .i10 = 0,
        .alpha = 1.0f,
        .limit = 0.0f,
    };
    NSUInteger nth = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger ds4_nth = n > 1 ? (NSUInteger)n / 2u : 1u;
    if (nth > ds4_nth) nth = ds4_nth;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_swiglu_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate offset:gate_off atIndex:1];
    [enc setBuffer:up   offset:up_off   atIndex:2];
    [enc setBuffer:out  offset:out_off  atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        mid,
        NSUInteger           mid_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows,
        float                clamp_value,
        bool                 mid_f16) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline(mid_f16 ? "kernel_dsv4_moe_swiglu_weight_f16" :
                                         "kernel_dsv4_moe_swiglu_weight");
    if (!pipeline) return 0;

    ds4_gpu_dsv4_moe_swiglu_weight_args args = {
        .width = width,
        .rows = rows,
        .gate_row_stride = (uint64_t)width * sizeof(float),
        .up_row_stride = (uint64_t)width * sizeof(float),
        .mid_row_stride = (uint64_t)width * (mid_f16 ? sizeof(uint16_t) : sizeof(float)),
        .weight_stride = sizeof(float),
        .write_clamped = getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ? 1u : 0u,
        .clamp_value = clamp_value,
    };

    NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate    offset:gate_off    atIndex:1];
    [enc setBuffer:up      offset:up_off      atIndex:2];
    [enc setBuffer:mid     offset:mid_off     atIndex:3];
    [enc setBuffer:weights offset:weights_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_sum6(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum6_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 6u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum6_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum6_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    if (n_expert == 6 &&
        ds4_gpu_encode_moe_sum6(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    ds4_gpu_bin_args first =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!ds4_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    ds4_gpu_bin_args accum =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!ds4_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}

static int ds4_gpu_encode_get_rows_i32_token_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        table,
        NSUInteger           table_off,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_off,
        const int32_t       *token_inline,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        uint32_t             hash_rows,
        uint32_t             n_cols,
        uint32_t             n_tokens) {
    if (!cb || !table || !selected || hash_rows == 0 || n_cols == 0 || n_tokens == 0) return 0;
    if (!tokens && !token_inline) return 0;

    const uint64_t table_row_bytes = (uint64_t)n_cols * sizeof(int32_t);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int64_t)n_cols,
        .ne00 = (int64_t)n_cols,
        .nb01 = table_row_bytes,
        .nb02 = (uint64_t)hash_rows * table_row_bytes,
        .nb03 = (uint64_t)hash_rows * table_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = table_row_bytes,
        .nb2 = (uint64_t)n_tokens * table_row_bytes,
        .nb3 = (uint64_t)n_tokens * table_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_cols;
    const NSUInteger max_threads = g_get_rows_i32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;
    const NSUInteger nw0 = ((NSUInteger)n_cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:table offset:table_off atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_off atIndex:2];
    } else {
        [enc setBytes:token_inline length:sizeof(*token_inline) atIndex:2];
    }
    [enc setBuffer:selected offset:selected_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_get_rows_f32_router_weights(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        probs,
        NSUInteger           probs_off,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             n_expert,
        uint32_t             n_expert_used,
        uint32_t             n_tokens) {
    if (!cb || !probs || !selected || !weights || n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;

    const uint64_t probs_token_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_row_bytes = (uint64_t)n_expert_used * sizeof(int32_t);
    const uint64_t weights_row_bytes = (uint64_t)n_expert_used * sizeof(float);
    ds4_gpu_get_rows_args args = {
        .ne00t = 1,
        .ne00 = 1,
        .nb01 = sizeof(float),
        .nb02 = probs_token_bytes,
        .nb03 = (uint64_t)n_tokens * probs_token_bytes,
        .ne10 = (int64_t)n_expert_used,
        .nb10 = sizeof(int32_t),
        .nb11 = selected_row_bytes,
        .nb12 = (uint64_t)n_tokens * selected_row_bytes,
        .nb1 = sizeof(float),
        .nb2 = weights_row_bytes,
        .nb3 = (uint64_t)n_tokens * weights_row_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:probs offset:probs_off atIndex:1];
    [enc setBuffer:selected offset:selected_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_expert_used, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || width == 0 || rows == 0) return 0;

    const uint64_t src_row_bytes = (uint64_t)width * sizeof(float);
    ds4_gpu_kargs_sum_rows args = {
        .ne00 = (int64_t)width,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)rows * src_row_bytes,
        .nb03 = (uint64_t)rows * src_row_bytes,
        .ne0 = 1,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = sizeof(float),
        .nb2 = (uint64_t)rows * sizeof(float),
        .nb3 = (uint64_t)rows * sizeof(float),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_sum_rows_f32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)args.ne00 && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)args.ne00) nth = (NSUInteger)args.ne00;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_sum_rows_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_router_select(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *selected,
        ds4_gpu_tensor     *weights,
        ds4_gpu_tensor     *probs,
        id<MTLBuffer>         logitsbuf,
        NSUInteger            logits_off,
        id<MTLBuffer>         biasbuf,
        NSUInteger            bias_off,
        id<MTLBuffer>         hashbuf,
        NSUInteger            hash_off,
        id<MTLBuffer>         tokensbuf,
        NSUInteger            tokens_off,
        const int32_t        *single_token,
        uint32_t              hash_rows,
        uint32_t              n_tokens,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale,
        bool                  has_bias,
        bool                  hash_mode) {
    id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
    id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
    id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
    const NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
    const NSUInteger weights_off = ds4_gpu_tensor_offset(weights);
    const NSUInteger probs_off = ds4_gpu_tensor_offset(probs);

    if (!cb || !selectedbuf || !weightsbuf || !probsbuf || !logitsbuf ||
        n_tokens == 0 || n_expert == 0 || n_expert_used == 0) return 0;

    const NSUInteger probs_bytes = (NSUInteger)n_tokens * (NSUInteger)n_expert * sizeof(float);
    const bool flash_router_fast_path =
        n_expert == 256u &&
        n_expert_used == 6u &&
        fabsf(expert_weight_scale - 1.5f) <= 1.0e-6f;

    int ok = 0;
    if (flash_router_fast_path &&
        !g_quality_mode && n_tokens == 1 &&
        getenv("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION") == NULL) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        id<MTLComputePipelineState> router_finalize_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_finalize_one_pipeline,
                                    "kernel_dsv4_router_finalize_one");
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!softplus_sqrt_pipeline || !router_finalize_pipeline || !router_weights_pipeline) return 0;

        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
        if (!ok) return 0;

        const bool use_token_buffer = single_token == NULL;
        ds4_gpu_dsv4_router_select_one_args args = {
            .has_bias = has_bias ? 1u : 0u,
            .hash_mode = hash_mode ? 1u : 0u,
            .use_token_buffer = use_token_buffer ? 1u : 0u,
            .token = single_token ? (uint32_t)*single_token : 0u,
            .hash_rows = hash_rows,
        };

        const float zero_f32 = 0.0f;
        const int32_t zero_i32 = 0;
        if ((has_bias && !biasbuf) ||
            (hash_mode && !hashbuf) ||
            (use_token_buffer && !tokensbuf)) {
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_finalize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:probsbuf offset:probs_off atIndex:1];
        if (has_bias) {
            [enc setBuffer:biasbuf offset:bias_off atIndex:2];
        } else {
            [enc setBytes:&zero_f32 length:sizeof(zero_f32) atIndex:2];
        }
        if (hash_mode) {
            [enc setBuffer:hashbuf offset:hash_off atIndex:3];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:3];
        }
        if (use_token_buffer) {
            [enc setBuffer:tokensbuf offset:tokens_off atIndex:4];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:4];
        }
        [enc setBuffer:selectedbuf offset:selected_off atIndex:5];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) + 256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    const NSUInteger sum_bytes = (NSUInteger)n_tokens * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_router_weight_sum_buffer,
                                         &g_router_weight_sum_bytes,
                                         sum_bytes,
                                         "ds4_router_weight_sum")) {
        return 0;
    }

    if (flash_router_fast_path && !g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        ok = softplus_sqrt_pipeline &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
    } else {
        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_softplus_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f) &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_sqrt_pipeline,
                                             probsbuf,
                                             probs_off,
                                             probsbuf,
                                             probs_off,
                                             n_expert,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f);
    }
    if (!ok) return 0;

    if (hash_mode) {
        ok = ds4_gpu_encode_get_rows_i32_token_rows(cb,
                                                      hashbuf,
                                                      hash_off,
                                                      tokensbuf,
                                                      tokens_off,
                                                      single_token,
                                                      selectedbuf,
                                                      selected_off,
                                                      hash_rows,
                                                      n_expert_used,
                                                      n_tokens);
    } else {
        ds4_gpu_tensor *score_tensor = probs;
        DS4MetalTensor *selection_view = nil;

        if (has_bias) {
            if (!biasbuf ||
                !ds4_gpu_ensure_scratch_buffer(&g_router_selection_buffer,
                                                 &g_router_selection_bytes,
                                                 probs_bytes,
                                                 "ds4_router_selection")) {
                return 0;
            }

            ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(n_expert, n_tokens, n_expert);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_add_pipeline,
                                               &add_args,
                                               probsbuf,
                                               probs_off,
                                               biasbuf,
                                               bias_off,
                                               g_router_selection_buffer,
                                               0);
            if (!ok) return 0;

            selection_view = [DS4MetalTensor new];
            selection_view.buffer = g_router_selection_buffer;
            selection_view.offset = 0;
            selection_view.bytes = probs_bytes;
            selection_view.owner = 0;
            score_tensor = (__bridge ds4_gpu_tensor *)selection_view;
        }

        ok = ds4_gpu_indexer_topk_tensor(selected, score_tensor, n_expert, n_tokens, n_expert_used) != 0;
    }
    if (!ok) return 0;

    if (flash_router_fast_path && !g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!router_weights_pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    ok = ds4_gpu_encode_get_rows_f32_router_weights(cb,
                                                      probsbuf,
                                                      probs_off,
                                                      selectedbuf,
                                                      selected_off,
                                                      weightsbuf,
                                                      weights_off,
                                                      n_expert,
                                                      n_expert_used,
                                                      n_tokens) &&
         ds4_gpu_encode_sum_rows_f32(cb,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       n_expert_used,
                                       n_tokens) &&
         ds4_gpu_encode_unary_f32_rows(cb,
                                         g_unary_clamp_pipeline,
                                         g_router_weight_sum_buffer,
                                         0,
                                         g_router_weight_sum_buffer,
                                         0,
                                         1,
                                         n_tokens,
                                         0,
                                         6.103515625e-5f,
                                         ds4_gpu_positive_infinity());
    if (!ok) return 0;

    ds4_gpu_bin_args div_args = ds4_gpu_make_bin_rowwise_scalar_args(n_expert_used, n_tokens);
    const float scale = expert_weight_scale;
    ds4_gpu_bin_args scale_args = ds4_gpu_make_bin_rows_args(n_expert_used, n_tokens, 1);

    ok = ds4_gpu_encode_bin_f32_rows(cb,
                                       g_bin_div_row_pipeline,
                                       &div_args,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       weightsbuf,
                                       weights_off);
    if (!ok) return 0;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
    [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:1];
    [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)scale_args.ne1,
                                          (NSUInteger)scale_args.ne2,
                                          (NSUInteger)scale_args.ne3)
         threadsPerThreadgroup:MTLSizeMake(ds4_gpu_bin_threads(n_expert_used, g_bin_mul_scalar_pipeline), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

/* Upload the reduced-expert original-id -> compact-slot LUT into a small resident
 * GPU buffer. n_layer * 256 int16 (~21 KiB for 43 layers). Called once at load
 * for a shrunken model; a full model never calls this so the translation kernel
 * stays a no-op. Replaces any previous LUT. */
int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!lut || n_layer == 0) return 0;
    @autoreleasepool {
        const NSUInteger bytes = (NSUInteger)n_layer * 256u * sizeof(int16_t);
        id<MTLBuffer> buf = [g_device newBufferWithBytes:lut
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        if (!buf) {
            fprintf(stderr, "ds4: failed to allocate Metal expert keep-map LUT (%llu bytes)\n",
                    (unsigned long long)bytes);
            return 0;
        }
        buf.label = @"ds4_expert_keep_lut";
        g_expert_keep_lut_buffer = buf;
        g_expert_keep_lut_layers = n_layer;
    }
    return 1;
}

/* Rewrite a routed-expert selection tensor from original ids (0..255) to the
 * compact slots of a shrunken model's expert tensors, in place. Runs between
 * router selection and the routed-MoE matvec. No-op (returns 1) when no keep-map
 * LUT has been set, so a full model is unaffected. The kernel maps both top-k and
 * first-3-layer hash selections (both live in the same `selected` tensor) and
 * clamps dropped/out-of-range experts to slot 0 so the matvec never indexes out
 * of bounds. */
int ds4_gpu_translate_expert_ids(
        ds4_gpu_tensor       *selected,
        uint32_t                layer,
        uint32_t                n_expert_used,
        uint32_t                n_tokens,
        uint32_t                n_total_expert) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_expert_keep_lut_buffer) return 1;   /* full model: nothing to translate */
    if (!selected || n_expert_used == 0 || n_tokens == 0 || n_total_expert == 0) return 0;
    if (layer >= g_expert_keep_lut_layers) {
        fprintf(stderr, "ds4: expert id translation layer %u out of LUT range %u\n",
                layer, g_expert_keep_lut_layers);
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t need = (uint64_t)n_tokens * n_expert_used * sizeof(int32_t);
        if (!selbuf || ds4_gpu_tensor_bytes(selected) < need) {
            fprintf(stderr, "ds4: Metal expert id translation received an undersized selection buffer\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_route_translate_pipeline,
                                    "kernel_dsv4_route_translate");
        if (!pipeline) return 0;

        struct {
            uint32_t layer;
            uint32_t n_expert_used;
            uint32_t n_tokens;
            uint32_t n_total_expert;
        } args = { layer, n_expert_used, n_tokens, n_total_expert };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:g_expert_keep_lut_buffer offset:0 atIndex:1];
        [enc setBuffer:selbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        const NSUInteger total = (NSUInteger)n_tokens * n_expert_used;
        NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > total) tg = total;
        if (tg == 0) tg = 1;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned, "expert id translation")) return 0;
    }
    return 1;
}

static void ds4_gpu_expert_router_note(int token, int hash_mode);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert_used == 0) return 0;
    if (hash_mode && token >= hash_rows) return 0;
    /* Decode-token note for the exact hash-layer staging (project.md P2.1). */
    ds4_gpu_expert_router_note((int)token, hash_mode ? 1 : 0);
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert_used * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal router select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
            biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
            if (!biasbuf) return 0;
            bias_set_offset = (NSUInteger)bias_inner;
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        const int32_t token_i32 = (int32_t)token;
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      nil,
                                                      0,
                                                      &token_i32,
                                                      hash_rows,
                                                      1,
                                                      n_expert,
                                                      n_expert_used,
                                                      expert_weight_scale,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map ||
        n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        id<MTLBuffer> tokensbuf = ds4_gpu_tensor_buffer(tokens);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf || !tokensbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_tokens * n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * n_expert_used * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_tokens * n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(tokens) < (uint64_t)n_tokens * sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal router batch select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
            biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
            if (!biasbuf) return 0;
            bias_set_offset = (NSUInteger)bias_inner;
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      tokensbuf,
                                                      ds4_gpu_tensor_offset(tokens),
                                                      NULL,
                                                      hash_rows,
                                                      n_tokens,
                                                      n_expert,
                                                      n_expert_used,
                                                      expert_weight_scale,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

/* A3 expert offload: setting DS4_METAL_EXPERT_OFFLOAD=1 keeps routed expert
 * mmap views out of the resident working set.  The original safe path then
 * CPU-gathers only the routed slots into small resident scratch buffers.  That
 * path is memory-safe, but for q2-full it also forces a command-buffer drain +
 * CPU readback/copy at every routed-MoE layer.  DS4_METAL_EXPERT_OFFLOAD_DIRECT=1
 * keeps the non-resident expert views but lets the existing GPU id-matvec read
 * selected expert rows directly from those mmap-backed views, avoiding the A3
 * CPU gather barrier. */
static int ds4_gpu_expert_offload_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_METAL_EXPERT_OFFLOAD");
        cached = (v && v[0] && !(v[0] == '0' && v[1] == '\0')) ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: DS4_METAL_EXPERT_OFFLOAD=1: routed expert mmap views stay non-resident "
                    "when the loader can split them.\n");
        }
    }
    return cached;
}

static int ds4_gpu_expert_offload_direct_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_OFFLOAD_DIRECT") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: DS4_METAL_EXPERT_OFFLOAD_DIRECT=1: q2 routed experts bypass A3 CPU-gather; "
                    "GPU reads selected rows directly from non-resident mmap views.\n");
        } else if (ds4_gpu_expert_offload_enabled()) {
            fprintf(stderr,
                    "ds4: q2 routed experts use A3 CPU-gather scratch; set "
                    "DS4_METAL_EXPERT_OFFLOAD_DIRECT=1 to avoid per-layer CPU gather barriers.\n");
        }
    }
    return cached;
}


#define DS4_METAL_EXPERT_PROFILE_MAX_LAYERS 128u
#define DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS 1024u
#define DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES 65536u

typedef struct {
    uint64_t requests;   /* Unique (layer, expert) requests after per-call compaction. */
    uint64_t misses;     /* Misses in the simulated LRU hot pool. */
} ds4_metal_expert_profile_slot;

typedef struct {
    uint64_t calls;
    uint64_t routed_picks;
    uint64_t unique_requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t actual_copy_bytes;
    uint64_t simulated_miss_bytes;
    double   copy_ms;
} ds4_metal_expert_profile_layer;

typedef struct {
    bool     used;
    uint32_t layer;
    uint32_t expert;
    uint64_t bytes;
    int32_t  prev;
    int32_t  next;
} ds4_metal_expert_profile_cache_entry;

static int g_expert_profile_enabled = -1;
static bool g_expert_profile_initialized;
static bool g_expert_profile_summary_registered;
static uint64_t g_expert_profile_cache_capacity_bytes;
static uint64_t g_expert_profile_cache_used_bytes;
static uint64_t g_expert_profile_cache_peak_bytes;
static uint64_t g_expert_profile_cache_slots;
static uint64_t g_expert_profile_calls;
static uint64_t g_expert_profile_routed_picks;
static uint64_t g_expert_profile_unique_requests;
static uint64_t g_expert_profile_hits;
static uint64_t g_expert_profile_misses;
static uint64_t g_expert_profile_actual_copy_bytes;
static uint64_t g_expert_profile_simulated_miss_bytes;
static double   g_expert_profile_copy_ms;
static uint32_t g_expert_profile_interval;
static uint32_t g_expert_profile_top;
static bool     g_expert_profile_all;
static bool     g_expert_profile_memory_seen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_expert_profile_layer_experts[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint64_t g_expert_profile_gate_bytes[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint64_t g_expert_profile_down_bytes[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint64_t g_expert_profile_slot_bytes[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static ds4_metal_expert_profile_layer g_expert_profile_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static ds4_metal_expert_profile_slot g_expert_profile_slot[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static int32_t g_expert_profile_cache_index[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static ds4_metal_expert_profile_cache_entry g_expert_profile_cache[DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES];
static int32_t g_expert_profile_cache_head = -1;
static int32_t g_expert_profile_cache_tail = -1;
static int32_t g_expert_profile_cache_free = -1;

static uint64_t ds4_gpu_env_u64(const char *name, uint64_t defval) {
    const char *v = getenv(name);
    if (!v || !v[0]) return defval;
    char *endp = NULL;
    unsigned long long parsed = strtoull(v, &endp, 10);
    return endp != v ? (uint64_t)parsed : defval;
}

static void ds4_gpu_expert_profile_print_live(const char *tag) {
    if (g_expert_profile_unique_requests == 0) return;
    const double hit_pct = 100.0 * (double)g_expert_profile_hits /
                           (double)g_expert_profile_unique_requests;
    const double saved_pct = g_expert_profile_actual_copy_bytes ?
        100.0 * (double)(g_expert_profile_actual_copy_bytes -
                         (g_expert_profile_simulated_miss_bytes < g_expert_profile_actual_copy_bytes ?
                          g_expert_profile_simulated_miss_bytes : g_expert_profile_actual_copy_bytes)) /
        (double)g_expert_profile_actual_copy_bytes : 0.0;
    fprintf(stderr,
            "ds4: expert-profile %s: calls=%llu picks=%llu unique=%llu "
            "sim_lru_hit=%.2f%% hits=%llu misses=%llu cache=%.2f/%.2f MiB peak=%.2f MiB "
            "actual_copy=%.2f GiB sim_miss=%.2f GiB saved=%.2f%% copy=%.3f ms\n",
            tag ? tag : "live",
            (unsigned long long)g_expert_profile_calls,
            (unsigned long long)g_expert_profile_routed_picks,
            (unsigned long long)g_expert_profile_unique_requests,
            hit_pct,
            (unsigned long long)g_expert_profile_hits,
            (unsigned long long)g_expert_profile_misses,
            (double)g_expert_profile_cache_used_bytes / (1024.0 * 1024.0),
            (double)g_expert_profile_cache_capacity_bytes / (1024.0 * 1024.0),
            (double)g_expert_profile_cache_peak_bytes / (1024.0 * 1024.0),
            (double)g_expert_profile_actual_copy_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)g_expert_profile_simulated_miss_bytes / (1024.0 * 1024.0 * 1024.0),
            saved_pct,
            g_expert_profile_copy_ms);
}

static void ds4_gpu_expert_profile_summary(void) {
    if (!g_expert_profile_initialized || g_expert_profile_unique_requests == 0) return;
    ds4_gpu_expert_profile_print_live("summary");

    const uint32_t top = g_expert_profile_all ? DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS : g_expert_profile_top;
    for (uint32_t il = 0; il < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; il++) {
        ds4_metal_expert_profile_layer *ls = &g_expert_profile_layer[il];
        if (ls->unique_requests == 0) continue;
        const uint64_t slot_bytes = g_expert_profile_slot_bytes[il];
        const double layer_hit = 100.0 * (double)ls->hits / (double)ls->unique_requests;
        fprintf(stderr,
                "ds4: expert-profile layer %02u: experts=%u slot=%.3f MiB "
                "(gate=%.3f up=%.3f down=%.3f) layer_experts=%.3f GiB "
                "calls=%llu picks=%llu unique=%llu hit=%.2f%% misses=%llu "
                "actual=%.3f GiB sim_miss=%.3f GiB copy=%.3f ms\n",
                il,
                g_expert_profile_layer_experts[il],
                (double)slot_bytes / (1024.0 * 1024.0),
                (double)g_expert_profile_gate_bytes[il] / (1024.0 * 1024.0),
                (double)g_expert_profile_gate_bytes[il] / (1024.0 * 1024.0),
                (double)g_expert_profile_down_bytes[il] / (1024.0 * 1024.0),
                (double)(slot_bytes * (uint64_t)g_expert_profile_layer_experts[il]) /
                    (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)ls->calls,
                (unsigned long long)ls->routed_picks,
                (unsigned long long)ls->unique_requests,
                layer_hit,
                (unsigned long long)ls->misses,
                (double)ls->actual_copy_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)ls->simulated_miss_bytes / (1024.0 * 1024.0 * 1024.0),
                ls->copy_ms);

        bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
        uint32_t printed_count = 0;
        while (printed_count < top) {
            int best = -1;
            uint64_t best_req = 0;
            for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
                if (printed[e]) continue;
                const uint64_t req = g_expert_profile_slot[il][e].requests;
                if (req > best_req) {
                    best_req = req;
                    best = (int)e;
                }
            }
            if (best < 0 || best_req == 0) break;
            printed[(uint32_t)best] = true;
            printed_count++;
            const uint64_t miss = g_expert_profile_slot[il][best].misses;
            const uint64_t hit = best_req - miss;
            const double hp = 100.0 * (double)hit / (double)best_req;
            const bool resident = g_expert_profile_cache_index[il][best] >= 0;
            fprintf(stderr,
                    "ds4: expert-profile   L%02u E%03u req=%llu hit=%llu miss=%llu "
                    "hit=%.2f%% actual=%.3f MiB sim_miss=%.3f MiB resident=%d\n",
                    il,
                    (uint32_t)best,
                    (unsigned long long)best_req,
                    (unsigned long long)hit,
                    (unsigned long long)miss,
                    hp,
                    (double)(best_req * slot_bytes) / (1024.0 * 1024.0),
                    (double)(miss * slot_bytes) / (1024.0 * 1024.0),
                    resident ? 1 : 0);
        }
    }
}

static void ds4_gpu_expert_profile_init(void) {
    if (g_expert_profile_initialized) return;
    g_expert_profile_initialized = true;

    for (uint32_t il = 0; il < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; il++) {
        for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
            g_expert_profile_cache_index[il][e] = -1;
        }
    }
    for (uint32_t i = 0; i < DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES; i++) {
        g_expert_profile_cache[i].next = (i + 1u < DS4_METAL_EXPERT_PROFILE_MAX_CACHE_ENTRIES) ?
                                         (int32_t)(i + 1u) : -1;
        g_expert_profile_cache[i].prev = -1;
    }
    g_expert_profile_cache_free = 0;

    const uint64_t cache_mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_PROFILE_CACHE_MB", 1024u);
    g_expert_profile_cache_capacity_bytes = cache_mb * 1024ull * 1024ull;
    g_expert_profile_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_PROFILE_INTERVAL", 512u);
    g_expert_profile_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_PROFILE_TOP", 8u);
    if (g_expert_profile_top == 0) g_expert_profile_top = 1;
    if (g_expert_profile_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        g_expert_profile_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
    }
    g_expert_profile_all = ds4_gpu_env_bool("DS4_METAL_EXPERT_PROFILE_ALL") > 0;

    if (!g_expert_profile_summary_registered) {
        atexit(ds4_gpu_expert_profile_summary);
        g_expert_profile_summary_registered = true;
    }
    fprintf(stderr,
            "ds4: expert-offload profiler enabled: simulated LRU cache %.2f MiB, "
            "top=%u%s, live interval=%u MoE calls. Note: this profiles hit rate; "
            "A3 still copies active experts to scratch today.\n",
            (double)g_expert_profile_cache_capacity_bytes / (1024.0 * 1024.0),
            g_expert_profile_top,
            g_expert_profile_all ? " (ALL nonzero experts at exit)" : "",
            g_expert_profile_interval);
}

static int ds4_gpu_expert_profile_is_enabled(void) {
    if (g_expert_profile_enabled < 0) {
        g_expert_profile_enabled = ds4_gpu_env_bool("DS4_METAL_EXPERT_OFFLOAD_PROFILE") > 0 ? 1 : 0;
        if (g_expert_profile_enabled) ds4_gpu_expert_profile_init();
    }
    return g_expert_profile_enabled;
}

static void ds4_gpu_expert_profile_cache_unlink(int32_t idx) {
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    if (e->prev >= 0) g_expert_profile_cache[e->prev].next = e->next;
    if (e->next >= 0) g_expert_profile_cache[e->next].prev = e->prev;
    if (g_expert_profile_cache_head == idx) g_expert_profile_cache_head = e->next;
    if (g_expert_profile_cache_tail == idx) g_expert_profile_cache_tail = e->prev;
    e->prev = -1;
    e->next = -1;
}

static void ds4_gpu_expert_profile_cache_link_head(int32_t idx) {
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    e->prev = -1;
    e->next = g_expert_profile_cache_head;
    if (g_expert_profile_cache_head >= 0) g_expert_profile_cache[g_expert_profile_cache_head].prev = idx;
    g_expert_profile_cache_head = idx;
    if (g_expert_profile_cache_tail < 0) g_expert_profile_cache_tail = idx;
}

static void ds4_gpu_expert_profile_cache_touch(int32_t idx) {
    if (idx == g_expert_profile_cache_head) return;
    ds4_gpu_expert_profile_cache_unlink(idx);
    ds4_gpu_expert_profile_cache_link_head(idx);
}

static void ds4_gpu_expert_profile_cache_evict_tail(void) {
    const int32_t idx = g_expert_profile_cache_tail;
    if (idx < 0) return;
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    ds4_gpu_expert_profile_cache_unlink(idx);
    if (e->layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        e->expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        g_expert_profile_cache_index[e->layer][e->expert] = -1;
    }
    if (g_expert_profile_cache_used_bytes >= e->bytes) {
        g_expert_profile_cache_used_bytes -= e->bytes;
    } else {
        g_expert_profile_cache_used_bytes = 0;
    }
    if (g_expert_profile_cache_slots > 0) g_expert_profile_cache_slots--;
    e->used = false;
    e->bytes = 0;
    e->next = g_expert_profile_cache_free;
    e->prev = -1;
    g_expert_profile_cache_free = idx;
}

static int32_t ds4_gpu_expert_profile_cache_alloc_entry(void) {
    while (g_expert_profile_cache_free < 0 && g_expert_profile_cache_tail >= 0) {
        ds4_gpu_expert_profile_cache_evict_tail();
    }
    if (g_expert_profile_cache_free < 0) return -1;
    const int32_t idx = g_expert_profile_cache_free;
    g_expert_profile_cache_free = g_expert_profile_cache[idx].next;
    g_expert_profile_cache[idx].next = -1;
    g_expert_profile_cache[idx].prev = -1;
    return idx;
}

static void ds4_gpu_expert_profile_cache_insert(uint32_t layer, uint32_t expert, uint64_t bytes) {
    if (g_expert_profile_cache_capacity_bytes == 0 || bytes == 0 ||
        bytes > g_expert_profile_cache_capacity_bytes) {
        return;
    }
    while (g_expert_profile_cache_tail >= 0 &&
           g_expert_profile_cache_used_bytes + bytes > g_expert_profile_cache_capacity_bytes) {
        ds4_gpu_expert_profile_cache_evict_tail();
    }
    if (g_expert_profile_cache_used_bytes + bytes > g_expert_profile_cache_capacity_bytes) return;
    const int32_t idx = ds4_gpu_expert_profile_cache_alloc_entry();
    if (idx < 0) return;
    ds4_metal_expert_profile_cache_entry *e = &g_expert_profile_cache[idx];
    e->used = true;
    e->layer = layer;
    e->expert = expert;
    e->bytes = bytes;
    ds4_gpu_expert_profile_cache_link_head(idx);
    g_expert_profile_cache_index[layer][expert] = idx;
    g_expert_profile_cache_used_bytes += bytes;
    g_expert_profile_cache_slots++;
    if (g_expert_profile_cache_used_bytes > g_expert_profile_cache_peak_bytes) {
        g_expert_profile_cache_peak_bytes = g_expert_profile_cache_used_bytes;
    }
}

static void ds4_gpu_expert_profile_record(
        uint32_t layer_index,
        const uint32_t *active_ids,
        uint32_t n_active,
        uint32_t n_total_expert,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t routed_picks,
        double copy_ms) {
    if (!ds4_gpu_expert_profile_is_enabled()) return;
    if (!active_ids || n_active == 0) return;
    if (layer_index >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;

    const uint64_t slot_bytes = 2ull * gate_expert_bytes + down_expert_bytes;
    if (!g_expert_profile_memory_seen[layer_index]) {
        g_expert_profile_memory_seen[layer_index] = true;
        g_expert_profile_layer_experts[layer_index] = n_total_expert;
        g_expert_profile_gate_bytes[layer_index] = gate_expert_bytes;
        g_expert_profile_down_bytes[layer_index] = down_expert_bytes;
        g_expert_profile_slot_bytes[layer_index] = slot_bytes;
        fprintf(stderr,
                "ds4: expert-profile memory layer %02u: expert_slots=%u "
                "one_expert=%.3f MiB (gate=%.3f up=%.3f down=%.3f), "
                "all_layer_experts=%.3f GiB\n",
                layer_index,
                n_total_expert,
                (double)slot_bytes / (1024.0 * 1024.0),
                (double)gate_expert_bytes / (1024.0 * 1024.0),
                (double)gate_expert_bytes / (1024.0 * 1024.0),
                (double)down_expert_bytes / (1024.0 * 1024.0),
                (double)(slot_bytes * (uint64_t)n_total_expert) /
                    (1024.0 * 1024.0 * 1024.0));
    }

    ds4_metal_expert_profile_layer *ls = &g_expert_profile_layer[layer_index];
    ls->calls++;
    ls->routed_picks += routed_picks;
    ls->actual_copy_bytes += (uint64_t)n_active * slot_bytes;
    ls->copy_ms += copy_ms;

    g_expert_profile_calls++;
    g_expert_profile_routed_picks += routed_picks;
    g_expert_profile_actual_copy_bytes += (uint64_t)n_active * slot_bytes;
    g_expert_profile_copy_ms += copy_ms;

    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t expert = active_ids[i];
        if (expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) continue;
        g_expert_profile_slot[layer_index][expert].requests++;
        ls->unique_requests++;
        g_expert_profile_unique_requests++;

        int32_t idx = g_expert_profile_cache_index[layer_index][expert];
        if (idx >= 0 && g_expert_profile_cache[idx].used) {
            ds4_gpu_expert_profile_cache_touch(idx);
            ls->hits++;
            g_expert_profile_hits++;
        } else {
            g_expert_profile_slot[layer_index][expert].misses++;
            ls->misses++;
            ls->simulated_miss_bytes += slot_bytes;
            g_expert_profile_misses++;
            g_expert_profile_simulated_miss_bytes += slot_bytes;
            ds4_gpu_expert_profile_cache_insert(layer_index, expert, slot_bytes);
        }
    }

    if (g_expert_profile_interval != 0 &&
        (g_expert_profile_calls % g_expert_profile_interval) == 0) {
        ds4_gpu_expert_profile_print_live("live");
    }
}


typedef struct {
    bool        used;
    bool        ready;
    bool        loading;
    bool        busy;
    bool        pinned;
    const void *model_map;
    uint32_t    layer;
    uint32_t    expert;
    int32_t     prev;
    int32_t     next;
    uint64_t    last_used;
} ds4_metal_expert_pool_entry;

typedef struct {
    bool        used;
    const void *model_map;
    uint32_t    layer;
    uint32_t    expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
    uint32_t    n_expert_total;
} ds4_metal_expert_pool_meta;

typedef struct {
    bool        used;
    const void *model_map;
    uint32_t    layer;
    uint32_t    expert;
} ds4_metal_expert_prefetch_req;

static int g_expert_pool_enabled = -1;
static bool g_expert_pool_init_attempted;
static bool g_expert_pool_summary_registered;
static id<MTLBuffer> g_expert_pool_gate;
static id<MTLBuffer> g_expert_pool_up;
static id<MTLBuffer> g_expert_pool_down;
static uint64_t g_expert_pool_budget_bytes;
static uint64_t g_expert_pool_gate_expert_bytes;
static uint64_t g_expert_pool_down_expert_bytes;
static uint64_t g_expert_pool_slot_bytes;
static uint32_t g_expert_pool_slots;
static ds4_metal_expert_pool_entry *g_expert_pool_entries;
static ds4_metal_expert_pool_meta g_expert_pool_meta[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static int32_t g_expert_pool_head = -1;
static int32_t g_expert_pool_tail = -1;
static uint64_t g_expert_pool_calls;
static uint64_t g_expert_pool_requests;
static uint64_t g_expert_pool_hits;
static uint64_t g_expert_pool_misses;
static uint64_t g_expert_pool_inflight_hits;
static uint64_t g_expert_pool_prefetches;
static uint64_t g_expert_pool_prefetch_hits;
static uint64_t g_expert_pool_prefetch_misses;
static uint64_t g_expert_pool_prefetch_drops;
static uint64_t g_expert_pool_prefetch_copied;
static uint64_t g_expert_pool_sync_waits;
static uint64_t g_expert_pool_fallbacks;
static uint64_t g_expert_pool_miss_copy_bytes;
static double   g_expert_pool_miss_copy_ms;
static double   g_expert_pool_wait_ms;
static double   g_expert_pool_prefetch_copy_ms;
static uint32_t g_expert_pool_interval;
static uint32_t g_expert_pool_layer_start;
static uint32_t g_expert_pool_layer_end;
static uint32_t g_expert_pool_lookahead;
static uint32_t g_expert_pool_prefetch_top;
static int      g_expert_pool_prefetch_self;
static int      g_expert_pool_prefetch_adjacent;
static int      g_expert_pool_wait_inflight;
static int      g_expert_pool_foreground_fill;
static int      g_expert_pool_warm_batch;
static int      g_expert_pool_prefetch_evict;
static int      g_expert_pool_hit_only;
static uint32_t g_expert_pool_admit_after;
static int      g_expert_pool_hotlock_enabled;
static uint32_t g_expert_pool_hotlock_top;
static uint64_t g_expert_pool_admission_skips;
static uint64_t g_expert_pool_clock;
static bool     g_expert_pool_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static bool     g_expert_pool_static_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static bool     g_expert_pool_dynamic_pinned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static uint32_t g_expert_pool_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_expert_pool_static_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_expert_pool_dynamic_pinned_per_layer[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_expert_pool_pinned_total;
static uint32_t g_expert_pool_static_pinned_total;
static uint32_t g_expert_pool_dynamic_pinned_total;
static uint64_t g_expert_pool_pinned_requests;
static uint64_t g_expert_pool_pinned_hits;
static uint64_t g_expert_pool_pinned_misses;
static uint64_t g_expert_pool_pinned_prefetches;
static uint64_t g_expert_pool_auto_pin_updates;
static bool     g_expert_pool_pinned_env_parsed;
static bool     g_expert_pool_pinned_layer_queued[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_expert_pool_auto_pin_top;
static uint32_t g_expert_pool_auto_pin_min_req;
static uint32_t g_expert_pool_auto_pin_interval;
static uint32_t g_expert_pool_pin_reserve;
static uint32_t g_expert_pool_hotlist_top;
static uint32_t g_expert_pool_hotlist_interval;
static uint64_t g_expert_pool_auto_pin_last_req[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
#define DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX 64u
static uint32_t g_expert_pool_last_active_n[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_expert_pool_last_active[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX];
static uint64_t g_expert_pool_hot_count[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static pthread_mutex_t g_expert_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_expert_pool_cv = PTHREAD_COND_INITIALIZER;
static bool g_expert_pool_prefetch_thread_started;
static bool g_expert_pool_prefetch_shutdown;
static bool g_expert_pool_cycle_guard_reported;
static ds4_metal_expert_prefetch_req *g_expert_pool_prefetch_queue;
static uint32_t g_expert_pool_prefetch_qcap;
static uint32_t g_expert_pool_prefetch_qhead;
static uint32_t g_expert_pool_prefetch_qtail;
static uint32_t g_expert_pool_prefetch_qcount;

static uint32_t ds4_gpu_expert_pool_min_layer_slots(void) {
    static int initialized;
    static uint32_t min_slots;
    if (!initialized) {
        /* A layer needs at least the six routed experts used by one token, but
         * in practice routing jitters across tokens.  Twelve slots/layer gives
         * real temporal LRU room while still fitting several layers in the
         * worker budget.  Override with 6 for maximum layer coverage. */
        min_slots = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_MIN_LAYER_SLOTS", 12u);
        if (min_slots < 6u) min_slots = 6u;
        if (min_slots > 4096u) min_slots = 4096u;
        initialized = 1;
    }
    return min_slots;
}

static uint32_t ds4_gpu_expert_pool_requested_layers(void) {
    return g_expert_pool_layer_end >= g_expert_pool_layer_start ?
        (g_expert_pool_layer_end - g_expert_pool_layer_start + 1u) : 1u;
}

static uint32_t ds4_gpu_expert_pool_served_layers(void) {
    const uint32_t requested = ds4_gpu_expert_pool_requested_layers();
    const uint32_t min_slots = ds4_gpu_expert_pool_min_layer_slots();
    if (g_expert_pool_slots < min_slots) return 0;
    uint32_t served = g_expert_pool_slots / min_slots;
    if (served > requested) served = requested;
    return served;
}

static int ds4_gpu_expert_pool_layer_served(uint32_t layer, uint32_t *cap_out) {
    const uint32_t served = ds4_gpu_expert_pool_served_layers();
    if (served == 0) return 0;
    const uint32_t requested = ds4_gpu_expert_pool_requested_layers();
    (void)requested;
    /* Cache the tail of the configured layer range by default.  This is the
     * output-side worker bottleneck in the q2 two-host topology. */
    const uint32_t served_end = g_expert_pool_layer_end;
    const uint32_t served_start = served_end + 1u >= served ?
        (served_end + 1u - served) : g_expert_pool_layer_start;
    if (layer < served_start || layer > served_end) return 0;
    const uint32_t idx = layer - served_start;
    const uint32_t base = g_expert_pool_slots / served;
    const uint32_t rem = g_expert_pool_slots % served;
    uint32_t cap = base + (idx < rem ? 1u : 0u);
    if (cap < 6u) return 0;
    if (cap_out) *cap_out = cap;
    return 1;
}

static int ds4_gpu_expert_pool_layer_allowed(uint32_t layer) {
    return layer >= g_expert_pool_layer_start && layer <= g_expert_pool_layer_end;
}

static int ds4_gpu_expert_pool_pin_sep(char c) {
    return c == ';' || c == '|' || c == '/' || c == '\n';
}

static int ds4_gpu_expert_pool_add_pin(uint32_t layer, uint32_t expert, bool dynamic) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS ||
        expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        return 0;
    }
    int added = 0;
    if (!g_expert_pool_pinned[layer][expert]) {
        g_expert_pool_pinned[layer][expert] = true;
        g_expert_pool_pinned_per_layer[layer]++;
        g_expert_pool_pinned_total++;
        g_expert_pool_pinned_layer_queued[layer] = false;
        added = 1;
    }
    if (g_expert_pool_entries) {
        for (uint32_t slot = 0; slot < g_expert_pool_slots; slot++) {
            ds4_metal_expert_pool_entry *entry = &g_expert_pool_entries[slot];
            if (entry->used && entry->layer == layer && entry->expert == expert) {
                entry->pinned = true;
            }
        }
    }
    if (dynamic) {
        if (!g_expert_pool_static_pinned[layer][expert] &&
            !g_expert_pool_dynamic_pinned[layer][expert]) {
            g_expert_pool_dynamic_pinned[layer][expert] = true;
            g_expert_pool_dynamic_pinned_per_layer[layer]++;
            g_expert_pool_dynamic_pinned_total++;
        }
    } else if (!g_expert_pool_static_pinned[layer][expert]) {
        if (g_expert_pool_dynamic_pinned[layer][expert]) {
            g_expert_pool_dynamic_pinned[layer][expert] = false;
            if (g_expert_pool_dynamic_pinned_per_layer[layer] > 0) {
                g_expert_pool_dynamic_pinned_per_layer[layer]--;
            }
            if (g_expert_pool_dynamic_pinned_total > 0) g_expert_pool_dynamic_pinned_total--;
        }
        g_expert_pool_static_pinned[layer][expert] = true;
        g_expert_pool_static_pinned_per_layer[layer]++;
        g_expert_pool_static_pinned_total++;
    }
    return added;
}

static void ds4_gpu_expert_pool_parse_pinned_env(void) {
    if (g_expert_pool_pinned_env_parsed) return;
    g_expert_pool_pinned_env_parsed = true;
    const char *spec = getenv("DS4_METAL_EXPERT_POOL_PINNED");
    if (!spec || !spec[0]) return;

    const char *p = spec;
    while (*p) {
        while (*p && (isspace((unsigned char)*p) || ds4_gpu_expert_pool_pin_sep(*p))) p++;
        if (!*p) break;
        if (*p == 'L' || *p == 'l') p++;
        char *endp = NULL;
        unsigned long layer_ul = strtoul(p, &endp, 10);
        if (endp == p) {
            while (*p && !ds4_gpu_expert_pool_pin_sep(*p)) p++;
            continue;
        }
        p = endp;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p != ':' && *p != '=') {
            while (*p && !ds4_gpu_expert_pool_pin_sep(*p)) p++;
            continue;
        }
        p++;

        while (*p) {
            while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
            if (ds4_gpu_expert_pool_pin_sep(*p) || !*p) break;
            if (*p == 'E' || *p == 'e') p++;
            endp = NULL;
            unsigned long expert_ul = strtoul(p, &endp, 10);
            if (endp == p) {
                while (*p && *p != ',' && !ds4_gpu_expert_pool_pin_sep(*p)) p++;
                continue;
            }
            p = endp;
            if (layer_ul < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                expert_ul < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                (void)ds4_gpu_expert_pool_add_pin((uint32_t)layer_ul,
                                                  (uint32_t)expert_ul,
                                                  false);
            }
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p == ',') {
                p++;
                continue;
            }
            if (ds4_gpu_expert_pool_pin_sep(*p) || !*p) break;
            p++;
        }
    }

    if (g_expert_pool_pinned_total != 0) {
        fprintf(stderr,
                "ds4: expert-pool pinned whitelist loaded: %u experts from DS4_METAL_EXPERT_POOL_PINNED\n",
                g_expert_pool_pinned_total);
    }
}

static int ds4_gpu_expert_pool_is_pinned(uint32_t layer, uint32_t expert) {
    return layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
           expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS &&
           g_expert_pool_pinned[layer][expert];
}

static int ds4_gpu_expert_pool_entry_ready(uint32_t layer, uint32_t expert, int *pinned_out) {
    bool ready = false;
    bool pinned = false;
    if (g_expert_pool_entries) {
        for (uint32_t slot = 0; slot < g_expert_pool_slots; slot++) {
            ds4_metal_expert_pool_entry *entry = &g_expert_pool_entries[slot];
            if (entry->used && entry->layer == layer && entry->expert == expert) {
                if (entry->ready) ready = true;
                if (entry->pinned) pinned = true;
            }
        }
    }
    if (pinned_out) *pinned_out = pinned ? 1 : 0;
    return ready ? 1 : 0;
}

static void ds4_gpu_expert_pool_print_hotlist(const char *tag, uint32_t top, bool include_pinned_empty) {
    if (top == 0) return;
    if (top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;

    for (uint32_t layer = 0; layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; layer++) {
        uint64_t layer_req = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            layer_req += g_expert_pool_hot_count[layer][expert];
        }
        if (layer_req == 0 && (!include_pinned_empty || g_expert_pool_pinned_per_layer[layer] == 0)) continue;

        uint32_t layer_cap = 0;
        const int served = ds4_gpu_expert_pool_layer_served(layer, &layer_cap);
        fprintf(stderr,
                "ds4: expert-pool layer %02u: tag=%s requests=%llu pinned=%u static=%u dynamic=%u served=%d cap=%u\n",
                layer,
                tag ? tag : "live",
                (unsigned long long)layer_req,
                g_expert_pool_pinned_per_layer[layer],
                g_expert_pool_static_pinned_per_layer[layer],
                g_expert_pool_dynamic_pinned_per_layer[layer],
                served,
                layer_cap);

        bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
        uint32_t printed_count = 0;
        while (printed_count < top) {
            int best = -1;
            uint64_t best_req = 0;
            for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
                if (printed[expert]) continue;
                const uint64_t req = g_expert_pool_hot_count[layer][expert];
                if (req > best_req) {
                    best_req = req;
                    best = (int)expert;
                }
            }
            if (best < 0 || best_req == 0) break;
            printed[(uint32_t)best] = true;
            printed_count++;
            int entry_pinned = 0;
            const int resident = ds4_gpu_expert_pool_entry_ready(layer, (uint32_t)best, &entry_pinned);
            fprintf(stderr,
                    "ds4: expert-pool   L%02u E%03u req=%llu pinned=%d static=%d dynamic=%d resident=%d entry_pin=%d\n",
                    layer,
                    (uint32_t)best,
                    (unsigned long long)best_req,
                    ds4_gpu_expert_pool_is_pinned(layer, (uint32_t)best),
                    g_expert_pool_static_pinned[layer][best] ? 1 : 0,
                    g_expert_pool_dynamic_pinned[layer][best] ? 1 : 0,
                    resident,
                    entry_pinned);
        }

        if (!include_pinned_empty) continue;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS && printed_count < top; expert++) {
            if (!g_expert_pool_pinned[layer][expert] || printed[expert]) continue;
            printed_count++;
            int entry_pinned = 0;
            const int resident = ds4_gpu_expert_pool_entry_ready(layer, expert, &entry_pinned);
            fprintf(stderr,
                    "ds4: expert-pool   L%02u E%03u req=0 pinned=1 static=%d dynamic=%d resident=%d entry_pin=%d\n",
                    layer,
                    expert,
                    g_expert_pool_static_pinned[layer][expert] ? 1 : 0,
                    g_expert_pool_dynamic_pinned[layer][expert] ? 1 : 0,
                    resident,
                    entry_pinned);
        }
    }
}

static int ds4_gpu_expert_pool_is_enabled(void) {
    if (g_expert_pool_enabled < 0) {
        const uint64_t mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_MB", 0);
        g_expert_pool_enabled = mb != 0 ? 1 : 0;
        if (g_expert_pool_enabled) {
            g_expert_pool_budget_bytes = mb * 1024ull * 1024ull;
            g_expert_pool_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_INTERVAL", 64u);
            g_expert_pool_layer_start = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_LAYER_START", 0u);
            g_expert_pool_layer_end = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_LAYER_END", 127u);
            if (g_expert_pool_layer_end < g_expert_pool_layer_start) {
                g_expert_pool_layer_end = g_expert_pool_layer_start;
            }
            /* A nonzero DS4_METAL_EXPERT_POOL_MB should mean a real foreground
             * LRU cache: every miss is admitted and the least-recently-used entry
             * is evicted only when the pool is full.  Predictor prefetch is an
             * explicit opt-in layer on top of that; defaulting it on caused the
             * async copier to churn the pool and hide whether LRU itself helps. */
            g_expert_pool_lookahead = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PREFETCH_LOOKAHEAD", 0u);
            if (g_expert_pool_lookahead > 16u) g_expert_pool_lookahead = 16u;
            g_expert_pool_prefetch_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PREFETCH_TOP", 0u);
            if (g_expert_pool_prefetch_top > 64u) g_expert_pool_prefetch_top = 64u;
            g_expert_pool_prefetch_self = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_PREFETCH_SELF") > 0;
            g_expert_pool_prefetch_adjacent = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_PREFETCH_ADJACENT") > 0;
            g_expert_pool_wait_inflight = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_WAIT_INFLIGHT") > 0;
            int fg_fill_env = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_FOREGROUND_FILL");
            g_expert_pool_foreground_fill = fg_fill_env < 0 ? 1 : (fg_fill_env > 0);
            g_expert_pool_warm_batch = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_WARM_BATCH") > 0;
            g_expert_pool_prefetch_evict = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_PREFETCH_EVICT") > 0;
            int hit_only_env = ds4_gpu_env_bool("DS4_METAL_EXPERT_POOL_HIT_ONLY");
            g_expert_pool_hit_only = hit_only_env < 0 ? 1 : (hit_only_env > 0);
            g_expert_pool_admit_after = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_ADMIT_AFTER", 1u);
            if (g_expert_pool_admit_after == 0) g_expert_pool_admit_after = 1u;
            g_expert_pool_hotlock_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_HOTLOCK_TOP", 0u);
            if (g_expert_pool_hotlock_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_pool_hotlock_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
            }
            g_expert_pool_hotlock_enabled = g_expert_pool_hotlock_top != 0 ? 1 : 0;
            g_expert_pool_auto_pin_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_AUTO_PIN_TOP", 0u);
            if (g_expert_pool_auto_pin_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_pool_auto_pin_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
            }
            g_expert_pool_auto_pin_min_req = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_AUTO_PIN_MIN_REQ", 3u);
            if (g_expert_pool_auto_pin_min_req == 0) g_expert_pool_auto_pin_min_req = 1u;
            g_expert_pool_auto_pin_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_AUTO_PIN_INTERVAL", 32u);
            g_expert_pool_pin_reserve = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PIN_RESERVE", 8u);
            g_expert_pool_hotlist_top = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_HOTLIST_TOP", 16u);
            if (g_expert_pool_hotlist_top > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_pool_hotlist_top = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
            }
            g_expert_pool_hotlist_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_HOTLIST_INTERVAL", g_expert_pool_interval);
            ds4_gpu_expert_pool_parse_pinned_env();
        }
    }
    return g_expert_pool_enabled;
}

static void ds4_gpu_expert_pool_print(const char *tag) {
    if (g_expert_pool_calls == 0 && g_expert_pool_requests == 0 &&
        g_expert_pool_prefetches == 0 && g_expert_pool_fallbacks == 0 &&
        g_expert_pool_admission_skips == 0) return;
    const double hit_pct = g_expert_pool_requests ?
        100.0 * (double)g_expert_pool_hits / (double)g_expert_pool_requests : 0.0;
    const double pinned_hit_pct = g_expert_pool_pinned_requests ?
        100.0 * (double)g_expert_pool_pinned_hits / (double)g_expert_pool_pinned_requests : 0.0;
    fprintf(stderr,
            "ds4: expert-pool %s: calls=%llu requests=%llu hit=%.2f%% hits=%llu "
            "misses=%llu inflight=%llu fallbacks=%llu admit_skip=%llu slots=%u slot=%.3f MiB budget=%.2f MiB "
            "layers=%u:%u pinned=%u static=%u dynamic=%u auto_pin=%u/%u/%u auto_updates=%llu "
            "pinned_req=%llu pinned_hit=%.2f%% pinned_hits=%llu pinned_miss=%llu pinned_pf=%llu "
            "prefetch=%llu pf_hit=%llu pf_miss=%llu pf_drop=%llu pf_copy=%llu "
            "miss_copy=%.2f GiB sync_copy=%.3f ms pf_copy_ms=%.3f wait=%llu/%.3f ms\n",
            tag ? tag : "live",
            (unsigned long long)g_expert_pool_calls,
            (unsigned long long)g_expert_pool_requests,
            hit_pct,
            (unsigned long long)g_expert_pool_hits,
            (unsigned long long)g_expert_pool_misses,
            (unsigned long long)g_expert_pool_inflight_hits,
            (unsigned long long)g_expert_pool_fallbacks,
            (unsigned long long)g_expert_pool_admission_skips,
            g_expert_pool_slots,
            (double)g_expert_pool_slot_bytes / (1024.0 * 1024.0),
            (double)g_expert_pool_budget_bytes / (1024.0 * 1024.0),
            g_expert_pool_layer_start,
            g_expert_pool_layer_end,
            g_expert_pool_pinned_total,
            g_expert_pool_static_pinned_total,
            g_expert_pool_dynamic_pinned_total,
            g_expert_pool_auto_pin_top,
            g_expert_pool_auto_pin_min_req,
            g_expert_pool_pin_reserve,
            (unsigned long long)g_expert_pool_auto_pin_updates,
            (unsigned long long)g_expert_pool_pinned_requests,
            pinned_hit_pct,
            (unsigned long long)g_expert_pool_pinned_hits,
            (unsigned long long)g_expert_pool_pinned_misses,
            (unsigned long long)g_expert_pool_pinned_prefetches,
            (unsigned long long)g_expert_pool_prefetches,
            (unsigned long long)g_expert_pool_prefetch_hits,
            (unsigned long long)g_expert_pool_prefetch_misses,
            (unsigned long long)g_expert_pool_prefetch_drops,
            (unsigned long long)g_expert_pool_prefetch_copied,
            (double)g_expert_pool_miss_copy_bytes / (1024.0 * 1024.0 * 1024.0),
            g_expert_pool_miss_copy_ms,
            g_expert_pool_prefetch_copy_ms,
            (unsigned long long)g_expert_pool_sync_waits,
            g_expert_pool_wait_ms);
}

static void ds4_gpu_expert_pool_summary(void) {
    ds4_gpu_expert_pool_print("summary");
    uint32_t top = g_expert_pool_hotlist_top != 0 ? g_expert_pool_hotlist_top : 16u;
    ds4_gpu_expert_pool_print_hotlist("summary", top, true);
}

static void ds4_gpu_expert_pool_lru_unlink(int32_t idx) {
    ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[idx];
    if (e->prev >= 0) g_expert_pool_entries[e->prev].next = e->next;
    if (e->next >= 0) g_expert_pool_entries[e->next].prev = e->prev;
    if (g_expert_pool_head == idx) g_expert_pool_head = e->next;
    if (g_expert_pool_tail == idx) g_expert_pool_tail = e->prev;
    e->prev = -1;
    e->next = -1;
}

static void ds4_gpu_expert_pool_lru_link_head(int32_t idx) {
    ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[idx];
    e->prev = -1;
    e->next = g_expert_pool_head;
    if (g_expert_pool_head >= 0) g_expert_pool_entries[g_expert_pool_head].prev = idx;
    g_expert_pool_head = idx;
    if (g_expert_pool_tail < 0) g_expert_pool_tail = idx;
}

static void ds4_gpu_expert_pool_lru_touch(int32_t idx) {
    if (idx == g_expert_pool_head) return;
    ds4_gpu_expert_pool_lru_unlink(idx);
    ds4_gpu_expert_pool_lru_link_head(idx);
}

static int32_t ds4_gpu_expert_pool_find(const void *model_map, uint32_t layer, uint32_t expert) {
    if (!g_expert_pool_entries) return -1;
    for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
        ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[i];
        if (e->used && e->model_map == model_map && e->layer == layer && e->expert == expert) {
            return (int32_t)i;
        }
    }
    return -1;
}

static void ds4_gpu_expert_pool_clear_busy(void) {
    if (!g_expert_pool_entries) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
        g_expert_pool_entries[i].busy = false;
    }
    pthread_cond_broadcast(&g_expert_pool_cv);
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static int ds4_gpu_expert_pool_copy_slot(
        int32_t slot,
        const ds4_metal_expert_pool_meta *meta,
        uint32_t expert) {
    if (slot < 0 || !meta || !meta->used || expert >= meta->n_expert_total) return 0;
    uint8_t *gate_dst_base = (uint8_t *)g_expert_pool_gate.contents;
    uint8_t *up_dst_base = (uint8_t *)g_expert_pool_up.contents;
    uint8_t *down_dst_base = (uint8_t *)g_expert_pool_down.contents;
    const uint8_t *map = (const uint8_t *)meta->model_map;
    if (!gate_dst_base || !up_dst_base || !down_dst_base || !map) return 0;

    const uint64_t gate_src = (uint64_t)expert * meta->gate_expert_bytes;
    const uint64_t down_src = (uint64_t)expert * meta->down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)(uint32_t)slot * meta->gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)(uint32_t)slot * meta->down_expert_bytes;
    memcpy(gate_dst_base + gate_dst,
           map + meta->gate_offset + gate_src,
           (size_t)meta->gate_expert_bytes);
    memcpy(up_dst_base + gate_dst,
           map + meta->up_offset + gate_src,
           (size_t)meta->gate_expert_bytes);
    memcpy(down_dst_base + down_dst,
           map + meta->down_offset + down_src,
           (size_t)meta->down_expert_bytes);
    return 1;
}

static void ds4_gpu_expert_pool_register_layer_meta(
        const void *model_map,
        uint32_t    layer,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS || !model_map || n_expert_total == 0) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    ds4_metal_expert_pool_meta *m = &g_expert_pool_meta[layer];
    if (!m->used) {
        m->used = true;
        m->model_map = model_map;
        m->layer = layer;
        m->gate_offset = gate_offset;
        m->up_offset = up_offset;
        m->down_offset = down_offset;
        m->gate_expert_bytes = gate_expert_bytes;
        m->down_expert_bytes = down_expert_bytes;
        m->n_expert_total = n_expert_total;
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static int ds4_gpu_expert_pool_enqueue_prefetch_unlocked(
        const void *model_map,
        uint32_t layer,
        uint32_t expert) {
    if (!g_expert_pool_prefetch_queue || g_expert_pool_prefetch_qcap == 0) return 0;
    if (!ds4_gpu_expert_pool_layer_allowed(layer)) return 0;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return 0;
    const ds4_metal_expert_pool_meta *m = &g_expert_pool_meta[layer];
    if (!m->used || m->model_map != model_map || expert >= m->n_expert_total) return 0;

    int32_t slot = ds4_gpu_expert_pool_find(model_map, layer, expert);
    if (slot >= 0) return 1;
    for (uint32_t i = 0, q = g_expert_pool_prefetch_qhead;
         i < g_expert_pool_prefetch_qcount;
         i++, q = (q + 1u) % g_expert_pool_prefetch_qcap) {
        ds4_metal_expert_prefetch_req *r = &g_expert_pool_prefetch_queue[q];
        if (r->used && r->model_map == model_map && r->layer == layer && r->expert == expert) {
            return 1;
        }
    }
    if (g_expert_pool_prefetch_qcount >= g_expert_pool_prefetch_qcap) {
        g_expert_pool_prefetch_drops++;
        return 0;
    }
    ds4_metal_expert_prefetch_req *r = NULL;
    if (ds4_gpu_expert_pool_is_pinned(layer, expert)) {
        g_expert_pool_prefetch_qhead = (g_expert_pool_prefetch_qhead + g_expert_pool_prefetch_qcap - 1u) %
                                      g_expert_pool_prefetch_qcap;
        r = &g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead];
    } else {
        r = &g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qtail];
        g_expert_pool_prefetch_qtail = (g_expert_pool_prefetch_qtail + 1u) % g_expert_pool_prefetch_qcap;
    }
    r->used = true;
    r->model_map = model_map;
    r->layer = layer;
    r->expert = expert;
    g_expert_pool_prefetch_qcount++;
    pthread_cond_signal(&g_expert_pool_cv);
    return 1;
}

static void ds4_gpu_expert_pool_predict_enqueue(
        const void *model_map,
        uint32_t layer,
        const uint32_t *active_ids,
        uint32_t n_active) {
    if (!model_map || !active_ids || n_active == 0 || g_expert_pool_lookahead == 0) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
        uint32_t keep = n_active;
        if (keep > DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX) keep = DS4_METAL_EXPERT_POOL_LAST_ACTIVE_MAX;
        for (uint32_t i = 0; i < keep; i++) {
            g_expert_pool_last_active[layer][i] = active_ids[i];
        }
        g_expert_pool_last_active_n[layer] = keep;
    }
    if (g_expert_pool_prefetch_self && g_expert_pool_prefetch_top != 0 && layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
        uint32_t self_n = n_active;
        if (self_n > g_expert_pool_prefetch_top) self_n = g_expert_pool_prefetch_top;
        for (uint32_t i = 0; i < self_n; i++) {
            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, active_ids[i]);
        }
    }
    for (uint32_t d = 1; d <= g_expert_pool_lookahead; d++) {
        const uint32_t next_layer = layer + d;
        if (next_layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) break;
        const ds4_metal_expert_pool_meta *m = &g_expert_pool_meta[next_layer];
        if (!m->used || m->model_map != model_map) continue;
        /* Predictor 1 (opt-in): adjacent-layer reuse is weak for this GGUF.  Keep
         * it available for experiments, but do not let it flood the queue by
         * default; the decode-critical predictor is same-layer temporal reuse
         * below. */
        if (g_expert_pool_prefetch_adjacent) {
            const uint32_t active_prefetch = g_expert_pool_prefetch_top != 0 &&
                                            n_active > g_expert_pool_prefetch_top ?
                                            g_expert_pool_prefetch_top : n_active;
            for (uint32_t i = 0; i < active_prefetch; i++) {
                (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, next_layer, active_ids[i]);
            }
        }
        /* Predictor 2: once a layer has fired, repeat its previous active set on
         * the next token. This is the useful decode case; it preserves the LRU
         * resident pool while moving likely misses to the background thread. */
        uint32_t prev_n = g_expert_pool_last_active_n[next_layer];
        if (g_expert_pool_prefetch_top != 0 && prev_n > g_expert_pool_prefetch_top) {
            prev_n = g_expert_pool_prefetch_top;
        }
        for (uint32_t i = 0; i < prev_n; i++) {
            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map,
                                                                next_layer,
                                                                g_expert_pool_last_active[next_layer][i]);
        }
        /* Predictor 3: measured hot experts per layer, updated by the existing
         * profiler counters. This is opt-in via PREFETCH_TOP/HOTLOCK and helps
         * when a few code/text experts dominate. */
        if (g_expert_pool_prefetch_top != 0) {
            bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
            for (uint32_t k = 0; k < g_expert_pool_prefetch_top; k++) {
                int best = -1;
                uint64_t best_req = 0;
                for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
                    if (printed[e]) continue;
                    const uint64_t req = g_expert_pool_hot_count[next_layer][e] +
                                         g_expert_profile_slot[next_layer][e].requests;
                    if (req > best_req) {
                        best_req = req;
                        best = (int)e;
                    }
                }
                if (best < 0 || best_req == 0) break;
                printed[(uint32_t)best] = true;
                (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, next_layer, (uint32_t)best);
            }
        }
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static bool ds4_gpu_expert_pool_protected(
        const ds4_metal_expert_pool_entry *e,
        const void *model_map,
        uint32_t layer,
        const uint32_t *active_ids,
        uint32_t n_active) {
    if (!e || !e->used) return false;
    if (e->busy || e->loading) return true;
    if (e->model_map != model_map) return false;
    if (e->pinned) return true;
    if (e->layer != layer) {
        /* Effective LRU is per served layer: layer A must never evict layer B,
         * otherwise a small global pool cycles through layers and can hit 0% even
         * when each individual layer has strong temporal locality. */
        return true;
    }
    for (uint32_t i = 0; i < n_active; i++) {
        if (active_ids[i] == e->expert) return true;
    }
    return false;
}

static int32_t ds4_gpu_expert_pool_victim(
        const void *model_map,
        uint32_t layer,
        uint32_t layer_cap,
        const uint32_t *active_ids,
        uint32_t n_active) {
    uint32_t layer_used = 0;
    int32_t free_slot = -1;
    for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
        ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[i];
        if (!e->used) {
            if (free_slot < 0) free_slot = (int32_t)i;
            continue;
        }
        if (e->model_map == model_map && e->layer == layer) layer_used++;
    }
    if (free_slot >= 0 && layer_used < layer_cap) return free_slot;

    for (int32_t idx = g_expert_pool_tail; idx >= 0; idx = g_expert_pool_entries[idx].prev) {
        if (!ds4_gpu_expert_pool_protected(&g_expert_pool_entries[idx], model_map, layer,
                                           active_ids, n_active)) {
            ds4_gpu_expert_pool_lru_unlink(idx);
            memset(&g_expert_pool_entries[idx], 0, sizeof(g_expert_pool_entries[idx]));
            g_expert_pool_entries[idx].prev = -1;
            g_expert_pool_entries[idx].next = -1;
            return idx;
        }
    }
    return -1;
}

static void ds4_gpu_expert_pool_queue_pinned_layer(
        const void *model_map,
        uint32_t layer) {
    if (!model_map || g_expert_pool_pinned_total == 0) return;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    if (g_expert_pool_pinned_layer_queued[layer]) return;

    pthread_mutex_lock(&g_expert_pool_mu);
    if (!g_expert_pool_pinned_layer_queued[layer]) {
        uint32_t queued = 0;
        uint32_t wanted = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            if (!g_expert_pool_pinned[layer][expert]) continue;
            wanted++;
            if (ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, expert)) {
                queued++;
            }
        }
        if (queued != 0) {
            g_expert_pool_pinned_prefetches += queued;
        }
        if (wanted != 0 && queued == wanted) {
            g_expert_pool_pinned_layer_queued[layer] = true;
        }
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static void ds4_gpu_expert_pool_auto_pin_layer(
        const void *model_map,
        uint32_t layer,
        uint32_t layer_cap) {
    if (!model_map || g_expert_pool_auto_pin_top == 0) return;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS || layer_cap <= 6u) return;

    uint64_t layer_req = 0;
    for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
        layer_req += g_expert_pool_hot_count[layer][expert];
    }
    if (layer_req < g_expert_pool_auto_pin_min_req) return;
    if (g_expert_pool_auto_pin_interval != 0 &&
        layer_req < g_expert_pool_auto_pin_last_req[layer] + g_expert_pool_auto_pin_interval) {
        return;
    }
    g_expert_pool_auto_pin_last_req[layer] = layer_req;

    uint32_t max_pin = g_expert_pool_auto_pin_top;
    uint32_t reserve = g_expert_pool_pin_reserve;
    if (reserve < 6u) reserve = 6u;
    if (layer_cap > reserve) {
        const uint32_t cap_limit = layer_cap - reserve;
        if (max_pin > cap_limit) max_pin = cap_limit;
    } else {
        max_pin = 0;
    }
    if (max_pin == 0 || g_expert_pool_pinned_per_layer[layer] >= max_pin) return;

    bool chosen[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
    uint32_t added = 0;
    while (g_expert_pool_pinned_per_layer[layer] < max_pin) {
        int best = -1;
        uint64_t best_req = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            if (chosen[expert] || g_expert_pool_pinned[layer][expert]) continue;
            const uint64_t req = g_expert_pool_hot_count[layer][expert];
            if (req >= (uint64_t)g_expert_pool_auto_pin_min_req && req > best_req) {
                best_req = req;
                best = (int)expert;
            }
        }
        if (best < 0) break;
        chosen[(uint32_t)best] = true;
        if (ds4_gpu_expert_pool_add_pin(layer, (uint32_t)best, true)) {
            added++;
        }
    }
    if (added != 0) {
        uint32_t queued = 0;
        for (uint32_t expert = 0; expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; expert++) {
            if (!g_expert_pool_pinned[layer][expert]) continue;
            if (ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, expert)) queued++;
        }
        if (queued != 0) g_expert_pool_pinned_prefetches += queued;
        g_expert_pool_auto_pin_updates++;
    }
}

static bool ds4_gpu_expert_pool_choose_hotlock(
        uint32_t layer,
        uint32_t *experts_out,
        uint32_t *n_out) {
    if (!g_expert_pool_hotlock_enabled || !experts_out || !n_out) return false;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return false;
    if (g_expert_pool_slot_bytes == 0 || g_expert_pool_slots == 0) return false;
    uint32_t max_keep = g_expert_pool_hotlock_top;
    if (max_keep > g_expert_pool_slots) max_keep = g_expert_pool_slots;
    if (max_keep > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) max_keep = DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS;
    bool printed[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS] = { false };
    uint32_t count = 0;
    while (count < max_keep) {
        int best = -1;
        uint64_t best_req = 0;
        for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
            if (printed[e]) continue;
            const uint64_t req = g_expert_pool_hot_count[layer][e] +
                                 g_expert_profile_slot[layer][e].requests;
            if (req > best_req) {
                best_req = req;
                best = (int)e;
            }
        }
        if (best < 0 || best_req == 0) break;
        printed[(uint32_t)best] = true;
        experts_out[count++] = (uint32_t)best;
    }
    *n_out = count;
    return count != 0;
}

static void ds4_gpu_expert_pool_maybe_hotlock_layer(
        const void *model_map,
        uint32_t layer) {
    if (!g_expert_pool_hotlock_enabled || !model_map) return;
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    uint32_t hot[DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
    uint32_t n_hot = 0;
    if (!ds4_gpu_expert_pool_choose_hotlock(layer, hot, &n_hot)) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    for (uint32_t i = 0; i < n_hot; i++) {
        (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer, hot[i]);
    }
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static void *ds4_gpu_expert_pool_prefetch_main(void *arg) {
    (void)arg;
    for (;;) {
        ds4_metal_expert_prefetch_req req = { 0 };
        ds4_metal_expert_pool_meta meta = { 0 };
        int32_t slot = -1;

        pthread_mutex_lock(&g_expert_pool_mu);
        while (!g_expert_pool_prefetch_shutdown && g_expert_pool_prefetch_qcount == 0) {
            pthread_cond_wait(&g_expert_pool_cv, &g_expert_pool_mu);
        }
        if (g_expert_pool_prefetch_shutdown && g_expert_pool_prefetch_qcount == 0) {
            pthread_mutex_unlock(&g_expert_pool_mu);
            break;
        }
        req = g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead];
        memset(&g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead], 0,
               sizeof(g_expert_pool_prefetch_queue[g_expert_pool_prefetch_qhead]));
        g_expert_pool_prefetch_qhead = (g_expert_pool_prefetch_qhead + 1u) % g_expert_pool_prefetch_qcap;
        g_expert_pool_prefetch_qcount--;
        g_expert_pool_prefetches++;

        slot = ds4_gpu_expert_pool_find(req.model_map, req.layer, req.expert);
        if (slot >= 0) {
            ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
            if (e->ready) {
                g_expert_pool_prefetch_hits++;
                e->last_used = ++g_expert_pool_clock;
                ds4_gpu_expert_pool_lru_touch(slot);
            } else {
                g_expert_pool_inflight_hits++;
            }
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        if (req.layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
            g_expert_pool_prefetch_drops++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        meta = g_expert_pool_meta[req.layer];
        if (!meta.used || meta.model_map != req.model_map || req.expert >= meta.n_expert_total) {
            g_expert_pool_prefetch_drops++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        if (!g_expert_pool_prefetch_evict) {
            for (uint32_t i = 0; i < g_expert_pool_slots; i++) {
                if (!g_expert_pool_entries[i].used) {
                    slot = (int32_t)i;
                    break;
                }
            }
        } else {
            uint32_t req_cap = 0;
            if (!ds4_gpu_expert_pool_layer_served(req.layer, &req_cap)) {
                g_expert_pool_prefetch_drops++;
                pthread_mutex_unlock(&g_expert_pool_mu);
                continue;
            }
            slot = ds4_gpu_expert_pool_victim(req.model_map, req.layer, req_cap, NULL, 0);
        }
        if (slot < 0) {
            g_expert_pool_prefetch_drops++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            continue;
        }
        ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
        e->used = true;
        e->ready = false;
        e->loading = true;
        e->busy = false;
        e->model_map = req.model_map;
        e->layer = req.layer;
        e->expert = req.expert;
        e->pinned = ds4_gpu_expert_pool_is_pinned(req.layer, req.expert);
        e->last_used = ++g_expert_pool_clock;
        ds4_gpu_expert_pool_lru_link_head(slot);
        g_expert_pool_prefetch_misses++;
        pthread_mutex_unlock(&g_expert_pool_mu);

        const double t0 = ds4_gpu_now_ms();
        const int ok = ds4_gpu_expert_pool_copy_slot(slot, &meta, req.expert);
        const double dt = ds4_gpu_now_ms() - t0;

        pthread_mutex_lock(&g_expert_pool_mu);
        e = &g_expert_pool_entries[slot];
        if (e->used && e->model_map == req.model_map && e->layer == req.layer &&
            e->expert == req.expert && e->loading) {
            if (ok) {
                e->ready = true;
                e->loading = false;
                g_expert_pool_prefetch_copied++;
                g_expert_pool_prefetch_copy_ms += dt;
            } else {
                ds4_gpu_expert_pool_lru_unlink(slot);
                memset(e, 0, sizeof(*e));
                e->prev = -1;
                e->next = -1;
                g_expert_pool_prefetch_drops++;
            }
            pthread_cond_broadcast(&g_expert_pool_cv);
        }
        pthread_mutex_unlock(&g_expert_pool_mu);
    }
    return NULL;
}

static void ds4_gpu_expert_pool_stop_prefetch(void) {
    if (!g_expert_pool_prefetch_thread_started) return;
    pthread_mutex_lock(&g_expert_pool_mu);
    g_expert_pool_prefetch_shutdown = true;
    pthread_cond_broadcast(&g_expert_pool_cv);
    pthread_mutex_unlock(&g_expert_pool_mu);
}

static int ds4_gpu_expert_pool_init(uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (!ds4_gpu_expert_pool_is_enabled()) return 0;
    const uint64_t slot_bytes = 2ull * gate_expert_bytes + down_expert_bytes;
    if (slot_bytes == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;

    if (g_expert_pool_slots != 0) {
        return g_expert_pool_gate_expert_bytes == gate_expert_bytes &&
               g_expert_pool_down_expert_bytes == down_expert_bytes;
    }
    if (g_expert_pool_init_attempted) return 0;
    g_expert_pool_init_attempted = true;

    uint64_t slots64 = g_expert_pool_budget_bytes / slot_bytes;
    if (slots64 == 0 || slots64 > (uint64_t)INT32_MAX) return 0;
    if (slots64 > (uint64_t)UINT32_MAX) slots64 = UINT32_MAX;
    const uint32_t slots = (uint32_t)slots64;
    const uint64_t gate_total = (uint64_t)slots * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)slots * down_expert_bytes;
    if (gate_total > NSUIntegerMax || down_total > NSUIntegerMax) return 0;

    @autoreleasepool {
        g_expert_pool_gate = [g_device newBufferWithLength:(NSUInteger)gate_total
                                                   options:MTLResourceStorageModeShared];
        g_expert_pool_up = [g_device newBufferWithLength:(NSUInteger)gate_total
                                                 options:MTLResourceStorageModeShared];
        g_expert_pool_down = [g_device newBufferWithLength:(NSUInteger)down_total
                                                   options:MTLResourceStorageModeShared];
        if (!g_expert_pool_gate || !g_expert_pool_up || !g_expert_pool_down) {
            fprintf(stderr,
                    "ds4: expert-pool failed to allocate %.2f MiB (%u slots); falling back to A3 scratch\n",
                    (double)(2ull * gate_total + down_total) / (1024.0 * 1024.0),
                    slots);
            g_expert_pool_gate = nil;
            g_expert_pool_up = nil;
            g_expert_pool_down = nil;
            return 0;
        }
        g_expert_pool_gate.label = @"ds4_expert_pool_gate";
        g_expert_pool_up.label = @"ds4_expert_pool_up";
        g_expert_pool_down.label = @"ds4_expert_pool_down";
    }

    g_expert_pool_entries = calloc((size_t)slots, sizeof(g_expert_pool_entries[0]));
    if (!g_expert_pool_entries) {
        g_expert_pool_gate = nil;
        g_expert_pool_up = nil;
        g_expert_pool_down = nil;
        return 0;
    }
    for (uint32_t i = 0; i < slots; i++) {
        g_expert_pool_entries[i].prev = -1;
        g_expert_pool_entries[i].next = -1;
    }
    g_expert_pool_gate_expert_bytes = gate_expert_bytes;
    g_expert_pool_down_expert_bytes = down_expert_bytes;
    g_expert_pool_slot_bytes = slot_bytes;
    g_expert_pool_slots = slots;
    if ((g_expert_pool_lookahead != 0 || g_expert_pool_pinned_total != 0 || g_expert_pool_hit_only) && !g_expert_pool_prefetch_queue) {
        uint64_t qcap64 = ds4_gpu_env_u64("DS4_METAL_EXPERT_POOL_PREFETCH_QUEUE", (uint64_t)slots * 8ull);
        if (qcap64 < 64u) qcap64 = 64u;
        if (qcap64 > 65536u) qcap64 = 65536u;
        g_expert_pool_prefetch_qcap = (uint32_t)qcap64;
        g_expert_pool_prefetch_queue = calloc(g_expert_pool_prefetch_qcap,
                                              sizeof(g_expert_pool_prefetch_queue[0]));
        if (g_expert_pool_prefetch_queue) {
            pthread_t th;
            if (pthread_create(&th, NULL, ds4_gpu_expert_pool_prefetch_main, NULL) == 0) {
                pthread_detach(th);
                g_expert_pool_prefetch_thread_started = true;
                atexit(ds4_gpu_expert_pool_stop_prefetch);
            } else {
                free(g_expert_pool_prefetch_queue);
                g_expert_pool_prefetch_queue = NULL;
                g_expert_pool_prefetch_qcap = 0;
                fprintf(stderr, "ds4: expert-pool async prefetch thread failed to start; continuing sync-only\n");
            }
        }
    }
    if (!g_expert_pool_summary_registered) {
        atexit(ds4_gpu_expert_pool_summary);
        g_expert_pool_summary_registered = true;
    }
    fprintf(stderr,
            "ds4: expert-pool enabled: %.2f MiB, %u slots, one expert %.3f MiB "
            "(gate=%.3f up=%.3f down=%.3f), configured layers %u:%u, serving tail %u/%u layers with >=%u slots/layer. "
            "LRU resident pool with async predictor prefetch lookahead=%u top=%u q=%u "
            "self=%s adjacent=%s admit_after=%u pinned=%u static=%u dynamic=%u auto_pin_top=%u min_req=%u reserve=%u hotlist=%u/%u hit_only=%s wait_inflight=%s foreground_fill=%s warm_batch=%s pf_evict=%s%s; hits bypass A3 scratch copy.\n",
            (double)(2ull * gate_total + down_total) / (1024.0 * 1024.0),
            slots,
            (double)slot_bytes / (1024.0 * 1024.0),
            (double)gate_expert_bytes / (1024.0 * 1024.0),
            (double)gate_expert_bytes / (1024.0 * 1024.0),
            (double)down_expert_bytes / (1024.0 * 1024.0),
            g_expert_pool_layer_start,
            g_expert_pool_layer_end,
            ds4_gpu_expert_pool_served_layers(),
            ds4_gpu_expert_pool_requested_layers(),
            ds4_gpu_expert_pool_min_layer_slots(),
            g_expert_pool_lookahead,
            g_expert_pool_prefetch_top,
            g_expert_pool_prefetch_qcap,
            g_expert_pool_prefetch_self ? "on" : "off",
            g_expert_pool_prefetch_adjacent ? "on" : "off",
            g_expert_pool_admit_after,
            g_expert_pool_pinned_total,
            g_expert_pool_static_pinned_total,
            g_expert_pool_dynamic_pinned_total,
            g_expert_pool_auto_pin_top,
            g_expert_pool_auto_pin_min_req,
            g_expert_pool_pin_reserve,
            g_expert_pool_hotlist_top,
            g_expert_pool_hotlist_interval,
            g_expert_pool_hit_only ? "on" : "off",
            g_expert_pool_wait_inflight ? "on" : "off",
            g_expert_pool_foreground_fill ? "on" : "off",
            g_expert_pool_warm_batch ? "on" : "off",
            g_expert_pool_prefetch_evict ? "on" : "off",
            g_expert_pool_hotlock_enabled ? " +hotlock" : "");
    return 1;
}

static int ds4_gpu_try_load_layer_experts_to_pool(
        const void *model_map,
        uint32_t    layer_index,
        id<MTLBuffer> selectedbuf,
        NSUInteger  selected_off,
        uint32_t    n_picks,
        uint32_t    n_active,
        const uint32_t *active_ids,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total,
        id<MTLBuffer> *gate_buf,
        id<MTLBuffer> *up_buf,
        id<MTLBuffer> *down_buf,
        uint32_t    *source_n_total_expert) {
    if (!ds4_gpu_expert_pool_is_enabled() || !ds4_gpu_expert_pool_layer_allowed(layer_index) ||
        !model_map || !selectedbuf || !active_ids ||
        !gate_buf || !up_buf || !down_buf || !source_n_total_expert || n_active == 0) {
        return 0;
    }
    if (!ds4_gpu_expert_pool_init(gate_expert_bytes, down_expert_bytes)) return 0;
    ds4_gpu_expert_pool_clear_busy();
    uint32_t layer_cap = 0;
    if (!ds4_gpu_expert_pool_layer_served(layer_index, &layer_cap)) {
        if (!g_expert_pool_cycle_guard_reported) {
            const uint32_t requested = ds4_gpu_expert_pool_requested_layers();
            const uint32_t served = ds4_gpu_expert_pool_served_layers();
            fprintf(stderr,
                    "ds4: expert-pool per-layer LRU: %u slots cannot serve layer %u in configured range %u:%u "
                    "(serving tail %u/%u layers, min_slots/layer=%u); bypassing this layer.\n",
                    g_expert_pool_slots,
                    layer_index,
                    g_expert_pool_layer_start,
                    g_expert_pool_layer_end,
                    served,
                    requested,
                    ds4_gpu_expert_pool_min_layer_slots());
            g_expert_pool_cycle_guard_reported = true;
        }
        g_expert_pool_fallbacks++;
        return 0;
    }
    if (n_active > layer_cap) {
        g_expert_pool_fallbacks++;
        return 0;
    }
    if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        g_expert_pool_pinned_per_layer[layer_index] > layer_cap) {
        static bool warned[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
        if (!warned[layer_index]) {
            fprintf(stderr,
                    "ds4: expert-pool pinned whitelist layer %u has %u experts but layer cap is %u; "
                    "LRU misses may fall back until DS4_METAL_EXPERT_POOL_MB or min_layer_slots is increased.\n",
                    layer_index,
                    g_expert_pool_pinned_per_layer[layer_index],
                    layer_cap);
            warned[layer_index] = true;
        }
    }
    if (selectedbuf.storageMode != MTLStorageModeShared) return 0;

    int32_t pool_slots[1024];
    if (n_active > 1024) return 0;
    if (!g_expert_pool_gate.contents || !g_expert_pool_up.contents || !g_expert_pool_down.contents) return 0;

    ds4_metal_expert_pool_meta meta = {
        .used = true,
        .model_map = model_map,
        .layer = layer_index,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .n_expert_total = n_expert_total,
    };
    ds4_gpu_expert_pool_register_layer_meta(model_map,
                                            layer_index,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes,
                                            n_expert_total);
    ds4_gpu_expert_pool_queue_pinned_layer(model_map, layer_index);

    if (g_expert_pool_hit_only) {
        bool all_ready = true;
        bool ready_flags[1024] = { false };
        pthread_mutex_lock(&g_expert_pool_mu);
        for (uint32_t i = 0; i < n_active; i++) {
            const uint32_t expert = active_ids[i];
            if (expert >= n_expert_total) {
                all_ready = false;
                continue;
            }
            const int32_t slot = ds4_gpu_expert_pool_find(model_map, layer_index, expert);
            ds4_metal_expert_pool_entry *entry = slot >= 0 ? &g_expert_pool_entries[slot] : NULL;
            ready_flags[i] = entry && entry->ready;
            if (!ready_flags[i]) {
                all_ready = false;
                if (g_expert_pool_prefetch_queue) {
                    (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map, layer_index, expert);
                }
            }
        }
        if (!all_ready) {
            for (uint32_t i = 0; i < n_active; i++) {
                const uint32_t expert = active_ids[i];
                if (expert >= n_expert_total) continue;
                const int request_pinned = ds4_gpu_expert_pool_is_pinned(layer_index, expert);
                g_expert_pool_requests++;
                if (request_pinned) g_expert_pool_pinned_requests++;
                if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                    expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                    g_expert_pool_hot_count[layer_index][expert]++;
                    ds4_gpu_expert_pool_auto_pin_layer(model_map, layer_index, layer_cap);
                }
                if (ready_flags[i]) {
                    const int32_t slot = ds4_gpu_expert_pool_find(model_map, layer_index, expert);
                    if (slot >= 0) {
                        g_expert_pool_hits++;
                        if (request_pinned) g_expert_pool_pinned_hits++;
                        g_expert_pool_entries[slot].last_used = ++g_expert_pool_clock;
                        ds4_gpu_expert_pool_lru_touch(slot);
                    }
                } else {
                    g_expert_pool_misses++;
                    if (request_pinned) g_expert_pool_pinned_misses++;
                }
            }
            g_expert_pool_admission_skips++;
            pthread_mutex_unlock(&g_expert_pool_mu);
            return 0;
        }
        pthread_mutex_unlock(&g_expert_pool_mu);
    }

    double copy_t0 = 0.0;
    bool copied_any = false;
    int32_t marked_busy[1024];
    uint32_t n_marked_busy = 0;

    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t expert = active_ids[i];
        if (expert >= n_expert_total) {
            ds4_gpu_expert_pool_clear_busy();
            return 0;
        }

        bool counted_request = false;
        bool request_pinned = false;
        for (;;) {
            int32_t slot = -1;
            bool need_sync_copy = false;
            bool wait_for_prefetch = false;

            pthread_mutex_lock(&g_expert_pool_mu);
            if (!counted_request) {
                g_expert_pool_requests++;
                const int was_pinned = ds4_gpu_expert_pool_is_pinned(layer_index, expert);
                if (was_pinned) {
                    g_expert_pool_pinned_requests++;
                    request_pinned = true;
                }
                if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                    expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                    g_expert_pool_hot_count[layer_index][expert]++;
                    ds4_gpu_expert_pool_auto_pin_layer(model_map, layer_index, layer_cap);
                }
                counted_request = true;
            }

            slot = ds4_gpu_expert_pool_find(model_map, layer_index, expert);
            if (slot >= 0) {
                ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                if (e->ready) {
                    g_expert_pool_hits++;
                    if (request_pinned) g_expert_pool_pinned_hits++;
                    e->busy = true;
                    e->last_used = ++g_expert_pool_clock;
                    ds4_gpu_expert_pool_lru_touch(slot);
                    pool_slots[i] = slot;
                    if (n_marked_busy < 1024u) marked_busy[n_marked_busy++] = slot;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    break;
                }
                if (e->loading) {
                    g_expert_pool_inflight_hits++;
                    if (g_expert_pool_wait_inflight) {
                        wait_for_prefetch = true;
                    } else {
                        pthread_mutex_unlock(&g_expert_pool_mu);
                        ds4_gpu_expert_pool_clear_busy();
                        return 0;
                    }
                } else {
                    /* Stale half-entry: evict and re-copy synchronously below. */
                    ds4_gpu_expert_pool_lru_unlink(slot);
                    memset(e, 0, sizeof(*e));
                    e->prev = -1;
                    e->next = -1;
                    slot = -1;
                }
            }

            if (wait_for_prefetch) {
                const double wait_t0 = ds4_gpu_now_ms();
                g_expert_pool_sync_waits++;
                while (slot >= 0) {
                    ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                    if (!(e->used && e->model_map == model_map && e->layer == layer_index &&
                          e->expert == expert && e->loading && !e->ready)) {
                        break;
                    }
                    pthread_cond_wait(&g_expert_pool_cv, &g_expert_pool_mu);
                }
                g_expert_pool_wait_ms += ds4_gpu_now_ms() - wait_t0;
                pthread_mutex_unlock(&g_expert_pool_mu);
                continue;
            }

            if (slot < 0) {
                if (!g_expert_pool_foreground_fill) {
                    if (g_expert_pool_prefetch_self && g_expert_pool_prefetch_top != 0) {
                        uint32_t self_n = n_active;
                        if (self_n > g_expert_pool_prefetch_top) self_n = g_expert_pool_prefetch_top;
                        for (uint32_t j = 0; j < self_n; j++) {
                            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map,
                                                                                layer_index,
                                                                                active_ids[j]);
                        }
                    }
                    g_expert_pool_admission_skips++;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    ds4_gpu_expert_pool_clear_busy();
                    return 0;
                }
                if (g_expert_pool_admit_after > 1u && layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                    expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS &&
                    g_expert_pool_hot_count[layer_index][expert] < g_expert_pool_admit_after) {
                    if (g_expert_pool_prefetch_self && g_expert_pool_prefetch_top != 0) {
                        uint32_t self_n = n_active;
                        if (self_n > g_expert_pool_prefetch_top) self_n = g_expert_pool_prefetch_top;
                        for (uint32_t j = 0; j < self_n; j++) {
                            (void)ds4_gpu_expert_pool_enqueue_prefetch_unlocked(model_map,
                                                                                layer_index,
                                                                                active_ids[j]);
                        }
                    }
                    for (uint32_t j = 0; j < n_active; j++) {
                        if (j == i) continue;
                        const uint32_t other = active_ids[j];
                        if (layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                            other < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                            g_expert_pool_requests++;
                            g_expert_pool_hot_count[layer_index][other]++;
                        }
                    }
                    g_expert_pool_admission_skips++;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    ds4_gpu_expert_pool_clear_busy();
                    return 0;
                }
                slot = ds4_gpu_expert_pool_victim(model_map, layer_index, layer_cap, active_ids, n_active);
                if (slot < 0) {
                    g_expert_pool_fallbacks++;
                    pthread_mutex_unlock(&g_expert_pool_mu);
                    ds4_gpu_expert_pool_clear_busy();
                    return 0;
                }
                ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                e->used = true;
                e->ready = false;
                e->loading = true;
                e->busy = false;
                e->model_map = model_map;
                e->layer = layer_index;
                e->expert = expert;
                e->pinned = ds4_gpu_expert_pool_is_pinned(layer_index, expert);
                e->last_used = ++g_expert_pool_clock;
                ds4_gpu_expert_pool_lru_link_head(slot);
                need_sync_copy = true;
                g_expert_pool_misses++;
                if (request_pinned) g_expert_pool_pinned_misses++;
                pthread_mutex_unlock(&g_expert_pool_mu);
            } else {
                pthread_mutex_unlock(&g_expert_pool_mu);
            }

            if (need_sync_copy) {
                if (!copied_any) {
                    copy_t0 = ds4_gpu_now_ms();
                    copied_any = true;
                }
                const double one_t0 = ds4_gpu_now_ms();
                const int ok = ds4_gpu_expert_pool_copy_slot(slot, &meta, expert);
                const double one_dt = ds4_gpu_now_ms() - one_t0;
                (void)one_dt;

                pthread_mutex_lock(&g_expert_pool_mu);
                ds4_metal_expert_pool_entry *e = &g_expert_pool_entries[slot];
                if (e->used && e->model_map == model_map && e->layer == layer_index &&
                    e->expert == expert && e->loading) {
                    if (ok) {
                        e->ready = true;
                        e->loading = false;
                        e->busy = true;
                        e->last_used = ++g_expert_pool_clock;
                        ds4_gpu_expert_pool_lru_touch(slot);
                        pool_slots[i] = slot;
                        if (n_marked_busy < 1024u) marked_busy[n_marked_busy++] = slot;
                        g_expert_pool_miss_copy_bytes += g_expert_pool_slot_bytes;
                        pthread_cond_broadcast(&g_expert_pool_cv);
                        pthread_mutex_unlock(&g_expert_pool_mu);
                        break;
                    } else {
                        ds4_gpu_expert_pool_lru_unlink(slot);
                        memset(e, 0, sizeof(*e));
                        e->prev = -1;
                        e->next = -1;
                        pthread_cond_broadcast(&g_expert_pool_cv);
                        pthread_mutex_unlock(&g_expert_pool_mu);
                        ds4_gpu_expert_pool_clear_busy();
                        return 0;
                    }
                }
                pthread_mutex_unlock(&g_expert_pool_mu);
                continue;
            }
        }
    }
    if (copied_any) g_expert_pool_miss_copy_ms += ds4_gpu_now_ms() - copy_t0;

    int32_t *sel_cpu = (int32_t *)((uint8_t *)selectedbuf.contents + (size_t)selected_off);
    for (uint32_t i = 0; i < n_picks; i++) {
        int32_t compact = sel_cpu[i];
        if (compact < 0 || (uint32_t)compact >= n_active) {
            ds4_gpu_expert_pool_clear_busy();
            return 0;
        }
        sel_cpu[i] = pool_slots[(uint32_t)compact];
    }

    *gate_buf = g_expert_pool_gate;
    *up_buf = g_expert_pool_up;
    *down_buf = g_expert_pool_down;
    *source_n_total_expert = g_expert_pool_slots;
    g_expert_pool_calls++;
    if (g_expert_pool_interval != 0 && (g_expert_pool_calls % g_expert_pool_interval) == 0) {
        ds4_gpu_expert_pool_print("live");
    }
    if (g_expert_pool_hotlist_interval != 0 &&
        g_expert_pool_hotlist_top != 0 &&
        (g_expert_pool_calls % g_expert_pool_hotlist_interval) == 0) {
        ds4_gpu_expert_pool_print_hotlist("live", g_expert_pool_hotlist_top, false);
    }

    /* Keep the LRU pool mandatory, but overlap future misses: current-layer active
     * IDs are a strong predictor for the next layer in this hash-routed q2 GGUF,
     * and hotlock (when enabled) keeps the measured top experts resident. */
    ds4_gpu_expert_pool_predict_enqueue(model_map, layer_index, active_ids, n_active);
    ds4_gpu_expert_pool_maybe_hotlock_layer(model_map, layer_index);
    (void)marked_busy;
    (void)n_marked_busy;
    return 1;
}

#define DS4_METAL_EXPERT_SOURCE_MAX_ENTRIES 65536u

typedef struct {
    bool     used;
    bool     hard_copy;
    bool     gate_locked;
    bool     up_locked;
    bool     down_locked;
    const void *model_map;
    uint32_t layer;
    uint32_t expert;
    uint64_t charged_bytes;
    void    *gate_base;
    void    *up_base;
    void    *down_base;
    size_t   gate_len;
    size_t   up_len;
    size_t   down_len;
    void    *gate_copy;
    void    *up_copy;
    void    *down_copy;
    size_t   gate_bytes;
    size_t   up_bytes;
    size_t   down_bytes;
    int32_t  prev;
    int32_t  next;
} ds4_metal_expert_source_entry;

static int g_expert_source_enabled = -1;
static bool g_expert_source_init_attempted;
static bool g_expert_source_summary_registered;
static bool g_expert_source_mlock_disabled;
static bool g_expert_source_mlock_reported;
static uint64_t g_expert_source_budget_bytes;
static uint64_t g_expert_source_used_bytes;
static uint64_t g_expert_source_peak_bytes;
static uint32_t g_expert_source_layer_start;
static uint32_t g_expert_source_layer_end;
static uint32_t g_expert_source_admit_after;
static uint32_t g_expert_source_interval;
static int g_expert_source_use_mlock;
static int g_expert_source_hard_copy;
static ds4_metal_expert_source_entry *g_expert_source_entries;
static uint32_t g_expert_source_cap;
static int32_t g_expert_source_head = -1;
static int32_t g_expert_source_tail = -1;
static int32_t g_expert_source_index[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static uint32_t g_expert_source_seen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
static uint64_t g_expert_source_requests;
static uint64_t g_expert_source_hits;
static uint64_t g_expert_source_misses;
static uint64_t g_expert_source_admissions;
static uint64_t g_expert_source_evictions;
static uint64_t g_expert_source_soft_entries;
static uint64_t g_expert_source_locked_entries;
static uint64_t g_expert_source_lock_failures;
static double   g_expert_source_admit_ms;
static int      g_expert_source_async_prefetch;

typedef struct {
    const void *model_map;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t layer;
    uint32_t expert;
} ds4_metal_expert_source_prefetch_req;

static pthread_mutex_t g_expert_source_pf_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_expert_source_pf_cv = PTHREAD_COND_INITIALIZER;
static pthread_t g_expert_source_pf_thread;
static int g_expert_source_pf_thread_started;
static int g_expert_source_pf_shutdown;
static ds4_metal_expert_source_prefetch_req *g_expert_source_pf_q;
static uint32_t g_expert_source_pf_qcap;
static uint32_t g_expert_source_pf_qhead;
static uint32_t g_expert_source_pf_qtail;
static uint32_t g_expert_source_pf_qcount;
static uint64_t g_expert_source_pf_enqueued;
static uint64_t g_expert_source_pf_dropped;
static uint64_t g_expert_source_pf_done;
static double   g_expert_source_pf_ms;

static int ds4_gpu_expert_source_is_enabled(void) {
    if (g_expert_source_enabled < 0) {
        const uint64_t mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_MB", 0);
        g_expert_source_enabled = mb != 0 ? 1 : 0;
        if (g_expert_source_enabled) {
            g_expert_source_budget_bytes = mb * 1024ull * 1024ull;
            g_expert_source_layer_start = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_START", 3u);
            g_expert_source_layer_end = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_LAYER_END", 127u);
            if (g_expert_source_layer_end < g_expert_source_layer_start) {
                g_expert_source_layer_end = g_expert_source_layer_start;
            }
            g_expert_source_admit_after = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_ADMIT_AFTER", 2u);
            if (g_expert_source_admit_after == 0) g_expert_source_admit_after = 1;
            g_expert_source_interval = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_INTERVAL", 64u);
            g_expert_source_use_mlock = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_MLOCK") > 0;
            g_expert_source_hard_copy = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_HARD_COPY") > 0;
            g_expert_source_async_prefetch = ds4_gpu_env_bool("DS4_METAL_EXPERT_SOURCE_CACHE_ASYNC") > 0;
        }
    }
    return g_expert_source_enabled;
}

static int ds4_gpu_expert_source_layer_allowed(uint32_t layer) {
    return layer >= g_expert_source_layer_start && layer <= g_expert_source_layer_end;
}

static void ds4_gpu_expert_source_print(const char *tag) {
    if (g_expert_source_requests == 0) return;
    const double hit_pct = 100.0 * (double)g_expert_source_hits / (double)g_expert_source_requests;
    fprintf(stderr,
            "ds4: expert-source-cache %s: requests=%llu hit=%.2f%% hits=%llu misses=%llu "
            "admit=%llu evict=%llu used=%.2f/%.2f MiB peak=%.2f MiB "
            "layers=%u:%u admit_after=%u mlock=%s hard_copy=%s async=%s locked=%llu soft=%llu lock_fail=%llu admit=%.3f ms pf=%llu/%llu drop=%llu pf_ms=%.3f\n",
            tag ? tag : "live",
            (unsigned long long)g_expert_source_requests,
            hit_pct,
            (unsigned long long)g_expert_source_hits,
            (unsigned long long)g_expert_source_misses,
            (unsigned long long)g_expert_source_admissions,
            (unsigned long long)g_expert_source_evictions,
            (double)g_expert_source_used_bytes / (1024.0 * 1024.0),
            (double)g_expert_source_budget_bytes / (1024.0 * 1024.0),
            (double)g_expert_source_peak_bytes / (1024.0 * 1024.0),
            g_expert_source_layer_start,
            g_expert_source_layer_end,
            g_expert_source_admit_after,
            (g_expert_source_use_mlock && !g_expert_source_mlock_disabled) ? "on" : "off",
            g_expert_source_hard_copy ? "on" : "off",
            g_expert_source_async_prefetch ? "on" : "off",
            (unsigned long long)g_expert_source_locked_entries,
            (unsigned long long)g_expert_source_soft_entries,
            (unsigned long long)g_expert_source_lock_failures,
            g_expert_source_admit_ms,
            (unsigned long long)g_expert_source_pf_done,
            (unsigned long long)g_expert_source_pf_enqueued,
            (unsigned long long)g_expert_source_pf_dropped,
            g_expert_source_pf_ms);
}

static void ds4_gpu_expert_source_summary(void) {
    ds4_gpu_expert_source_print("summary");
}

static size_t ds4_gpu_page_size(void) {
    static size_t page;
    if (page == 0) {
        long p = sysconf(_SC_PAGESIZE);
        page = p > 0 ? (size_t)p : (size_t)4096;
    }
    return page;
}

static void ds4_gpu_expert_source_align_range(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        void **base_out,
        size_t *len_out) {
    const size_t page = ds4_gpu_page_size();
    const uintptr_t addr = (uintptr_t)model_map + (uintptr_t)offset;
    const uintptr_t start = addr & ~((uintptr_t)page - 1u);
    const uintptr_t end = (addr + (uintptr_t)bytes + (uintptr_t)page - 1u) & ~((uintptr_t)page - 1u);
    *base_out = (void *)start;
    *len_out = (size_t)(end - start);
}

static void ds4_gpu_expert_source_lru_unlink(int32_t idx) {
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    if (e->prev >= 0) g_expert_source_entries[e->prev].next = e->next;
    if (e->next >= 0) g_expert_source_entries[e->next].prev = e->prev;
    if (g_expert_source_head == idx) g_expert_source_head = e->next;
    if (g_expert_source_tail == idx) g_expert_source_tail = e->prev;
    e->prev = -1;
    e->next = -1;
}

static void ds4_gpu_expert_source_lru_link_head(int32_t idx) {
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    e->prev = -1;
    e->next = g_expert_source_head;
    if (g_expert_source_head >= 0) g_expert_source_entries[g_expert_source_head].prev = idx;
    g_expert_source_head = idx;
    if (g_expert_source_tail < 0) g_expert_source_tail = idx;
}

static void ds4_gpu_expert_source_lru_touch(int32_t idx) {
    if (idx == g_expert_source_head) return;
    ds4_gpu_expert_source_lru_unlink(idx);
    ds4_gpu_expert_source_lru_link_head(idx);
}

static void ds4_gpu_expert_source_unlock_entry(ds4_metal_expert_source_entry *e) {
    if (!e) return;
    if (e->gate_locked) (void)munlock(e->gate_base, e->gate_len);
    if (e->up_locked) (void)munlock(e->up_base, e->up_len);
    if (e->down_locked) (void)munlock(e->down_base, e->down_len);
    free(e->gate_copy);
    free(e->up_copy);
    free(e->down_copy);
    e->gate_copy = e->up_copy = e->down_copy = NULL;
    e->gate_locked = e->up_locked = e->down_locked = false;
}

static void ds4_gpu_expert_source_evict_tail(void) {
    const int32_t idx = g_expert_source_tail;
    if (idx < 0) return;
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    ds4_gpu_expert_source_lru_unlink(idx);
    if (e->layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        e->expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS &&
        g_expert_source_index[e->layer][e->expert] == idx) {
        g_expert_source_index[e->layer][e->expert] = -1;
    }
    ds4_gpu_expert_source_unlock_entry(e);
    if (g_expert_source_used_bytes >= e->charged_bytes) {
        g_expert_source_used_bytes -= e->charged_bytes;
    } else {
        g_expert_source_used_bytes = 0;
    }
    memset(e, 0, sizeof(*e));
    e->prev = -1;
    e->next = -1;
    g_expert_source_evictions++;
}

static int32_t ds4_gpu_expert_source_find(const void *model_map, uint32_t layer, uint32_t expert) {
    if (!g_expert_source_entries) return -1;
    if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        const int32_t idx = g_expert_source_index[layer][expert];
        if (idx >= 0 && (uint32_t)idx < g_expert_source_cap) {
            ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
            if (e->used && e->model_map == model_map && e->layer == layer && e->expert == expert) {
                return idx;
            }
        }
    }
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        ds4_metal_expert_source_entry *e = &g_expert_source_entries[i];
        if (e->used && e->model_map == model_map && e->layer == layer && e->expert == expert) {
            if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
                expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
                g_expert_source_index[layer][expert] = (int32_t)i;
            }
            return (int32_t)i;
        }
    }
    return -1;
}

static int32_t ds4_gpu_expert_source_free_entry(void) {
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        if (!g_expert_source_entries[i].used) return (int32_t)i;
    }
    ds4_gpu_expert_source_evict_tail();
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        if (!g_expert_source_entries[i].used) return (int32_t)i;
    }
    return -1;
}

static int ds4_gpu_expert_source_init(void) {
    if (!ds4_gpu_expert_source_is_enabled()) return 0;
    if (g_expert_source_entries) return 1;
    if (g_expert_source_init_attempted) return 0;
    g_expert_source_init_attempted = true;

    g_expert_source_cap = DS4_METAL_EXPERT_SOURCE_MAX_ENTRIES;
    g_expert_source_entries = calloc(g_expert_source_cap, sizeof(g_expert_source_entries[0]));
    if (!g_expert_source_entries) return 0;
    for (uint32_t l = 0; l < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; l++) {
        for (uint32_t e = 0; e < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS; e++) {
            g_expert_source_index[l][e] = -1;
        }
    }
    for (uint32_t i = 0; i < g_expert_source_cap; i++) {
        g_expert_source_entries[i].prev = -1;
        g_expert_source_entries[i].next = -1;
    }
    if (!g_expert_source_summary_registered) {
        atexit(ds4_gpu_expert_source_summary);
        g_expert_source_summary_registered = true;
    }
    fprintf(stderr,
            "ds4: expert-source-cache enabled: %.2f MiB %s LRU, layers %u:%u, "
            "admit_after=%u, mlock=%s. GPU still reads compact A3 scratch; cache keeps expert source hot.\n",
            (double)g_expert_source_budget_bytes / (1024.0 * 1024.0),
            g_expert_source_hard_copy ? "hard-copy" : "source-page",
            g_expert_source_layer_start,
            g_expert_source_layer_end,
            g_expert_source_admit_after,
            g_expert_source_use_mlock ? "on" : "off");
    return 1;
}

static int ds4_gpu_expert_source_try_mlock(void *base, size_t len, bool *locked) {
    *locked = false;
    if (!g_expert_source_use_mlock || g_expert_source_mlock_disabled) return 1;
    if (mlock(base, len) == 0) {
        *locked = true;
        return 1;
    }
    g_expert_source_lock_failures++;
    if (!g_expert_source_mlock_reported) {
        fprintf(stderr,
                "ds4: expert-source-cache mlock failed (%s); continuing with soft madvise/page-cache LRU\n",
                strerror(errno));
        g_expert_source_mlock_reported = true;
    }
    /* macOS often has a small mlock rlimit. Disable further attempts after the
     * first failure so hot-loop admission does not repeatedly pay syscall costs. */
    g_expert_source_mlock_disabled = true;
    return 0;
}

static void *ds4_gpu_expert_source_prefetch_main(void *arg) {
    (void)arg;
    for (;;) {
        ds4_metal_expert_source_prefetch_req req = { 0 };
        pthread_mutex_lock(&g_expert_source_pf_mu);
        while (!g_expert_source_pf_shutdown && g_expert_source_pf_qcount == 0) {
            pthread_cond_wait(&g_expert_source_pf_cv, &g_expert_source_pf_mu);
        }
        if (g_expert_source_pf_shutdown && g_expert_source_pf_qcount == 0) {
            pthread_mutex_unlock(&g_expert_source_pf_mu);
            break;
        }
        req = g_expert_source_pf_q[g_expert_source_pf_qhead];
        g_expert_source_pf_qhead = (g_expert_source_pf_qhead + 1u) % g_expert_source_pf_qcap;
        g_expert_source_pf_qcount--;
        pthread_mutex_unlock(&g_expert_source_pf_mu);

        void *gate_base = NULL, *up_base = NULL, *down_base = NULL;
        size_t gate_len = 0, up_len = 0, down_len = 0;
        ds4_gpu_expert_source_align_range(req.model_map,
                                          req.gate_offset + (uint64_t)req.expert * req.gate_expert_bytes,
                                          req.gate_expert_bytes,
                                          &gate_base,
                                          &gate_len);
        ds4_gpu_expert_source_align_range(req.model_map,
                                          req.up_offset + (uint64_t)req.expert * req.gate_expert_bytes,
                                          req.gate_expert_bytes,
                                          &up_base,
                                          &up_len);
        ds4_gpu_expert_source_align_range(req.model_map,
                                          req.down_offset + (uint64_t)req.expert * req.down_expert_bytes,
                                          req.down_expert_bytes,
                                          &down_base,
                                          &down_len);
        const double t0 = ds4_gpu_now_ms();
        (void)madvise(gate_base, gate_len, MADV_WILLNEED);
        (void)madvise(up_base, up_len, MADV_WILLNEED);
        (void)madvise(down_base, down_len, MADV_WILLNEED);
        const double dt = ds4_gpu_now_ms() - t0;
        __sync_fetch_and_add(&g_expert_source_pf_done, 1ull);
        g_expert_source_pf_ms += dt;
    }
    return NULL;
}

static void ds4_gpu_expert_source_prefetch_summary(void) {
    pthread_mutex_lock(&g_expert_source_pf_mu);
    g_expert_source_pf_shutdown = 1;
    pthread_cond_broadcast(&g_expert_source_pf_cv);
    pthread_mutex_unlock(&g_expert_source_pf_mu);
}

static int ds4_gpu_expert_source_prefetch_init(void) {
    if (!g_expert_source_async_prefetch) return 0;
    if (g_expert_source_pf_thread_started) return 1;
    g_expert_source_pf_qcap = (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_SOURCE_CACHE_ASYNC_QUEUE", 8192u);
    if (g_expert_source_pf_qcap < 64u) g_expert_source_pf_qcap = 64u;
    g_expert_source_pf_q = calloc(g_expert_source_pf_qcap, sizeof(g_expert_source_pf_q[0]));
    if (!g_expert_source_pf_q) return 0;
    if (pthread_create(&g_expert_source_pf_thread, NULL, ds4_gpu_expert_source_prefetch_main, NULL) != 0) {
        free(g_expert_source_pf_q);
        g_expert_source_pf_q = NULL;
        return 0;
    }
    pthread_detach(g_expert_source_pf_thread);
    g_expert_source_pf_thread_started = 1;
    atexit(ds4_gpu_expert_source_prefetch_summary);
    return 1;
}

static void ds4_gpu_expert_source_prefetch_enqueue(
        const void *model_map,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!model_map || !ds4_gpu_expert_source_prefetch_init()) return;
    ds4_metal_expert_source_prefetch_req req = {
        .model_map = model_map,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .expert = expert,
    };
    pthread_mutex_lock(&g_expert_source_pf_mu);
    if (g_expert_source_pf_qcount == g_expert_source_pf_qcap) {
        g_expert_source_pf_dropped++;
    } else {
        g_expert_source_pf_q[g_expert_source_pf_qtail] = req;
        g_expert_source_pf_qtail = (g_expert_source_pf_qtail + 1u) % g_expert_source_pf_qcap;
        g_expert_source_pf_qcount++;
        g_expert_source_pf_enqueued++;
        pthread_cond_signal(&g_expert_source_pf_cv);
    }
    pthread_mutex_unlock(&g_expert_source_pf_mu);
}

static void ds4_gpu_expert_source_admit(
        const void *model_map,
        uint32_t layer,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_expert_source_entries || expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) return;

    void *gate_base = NULL, *up_base = NULL, *down_base = NULL;
    size_t gate_len = 0, up_len = 0, down_len = 0;
    ds4_gpu_expert_source_align_range(model_map,
                                      gate_offset + (uint64_t)expert * gate_expert_bytes,
                                      gate_expert_bytes,
                                      &gate_base,
                                      &gate_len);
    ds4_gpu_expert_source_align_range(model_map,
                                      up_offset + (uint64_t)expert * gate_expert_bytes,
                                      gate_expert_bytes,
                                      &up_base,
                                      &up_len);
    ds4_gpu_expert_source_align_range(model_map,
                                      down_offset + (uint64_t)expert * down_expert_bytes,
                                      down_expert_bytes,
                                      &down_base,
                                      &down_len);
    const uint64_t charge = g_expert_source_hard_copy ?
        (gate_expert_bytes + gate_expert_bytes + down_expert_bytes) :
        ((uint64_t)gate_len + (uint64_t)up_len + (uint64_t)down_len);
    if (charge == 0 || charge > g_expert_source_budget_bytes) return;

    while (g_expert_source_tail >= 0 && g_expert_source_used_bytes + charge > g_expert_source_budget_bytes) {
        ds4_gpu_expert_source_evict_tail();
    }
    if (g_expert_source_used_bytes + charge > g_expert_source_budget_bytes) return;

    const int32_t idx = ds4_gpu_expert_source_free_entry();
    if (idx < 0) return;
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];

    const double t0 = ds4_gpu_now_ms();
    (void)madvise(gate_base, gate_len, MADV_WILLNEED);
    (void)madvise(up_base, up_len, MADV_WILLNEED);
    (void)madvise(down_base, down_len, MADV_WILLNEED);
    if (g_expert_source_hard_copy) {
        uint8_t *gcopy = (uint8_t *)malloc((size_t)gate_expert_bytes);
        uint8_t *ucopy = (uint8_t *)malloc((size_t)gate_expert_bytes);
        uint8_t *dcopy = (uint8_t *)malloc((size_t)down_expert_bytes);
        if (!gcopy || !ucopy || !dcopy) {
            free(gcopy); free(ucopy); free(dcopy);
            return;
        }
        const uint8_t *map = (const uint8_t *)model_map;
        memcpy(gcopy, map + gate_offset + (uint64_t)expert * gate_expert_bytes, (size_t)gate_expert_bytes);
        memcpy(ucopy, map + up_offset + (uint64_t)expert * gate_expert_bytes, (size_t)gate_expert_bytes);
        memcpy(dcopy, map + down_offset + (uint64_t)expert * down_expert_bytes, (size_t)down_expert_bytes);
        e->hard_copy = true;
        e->gate_copy = gcopy;
        e->up_copy = ucopy;
        e->down_copy = dcopy;
        e->gate_bytes = (size_t)gate_expert_bytes;
        e->up_bytes = (size_t)gate_expert_bytes;
        e->down_bytes = (size_t)down_expert_bytes;
    }
    bool gl = false, ul = false, dl = false;
    (void)ds4_gpu_expert_source_try_mlock(gate_base, gate_len, &gl);
    (void)ds4_gpu_expert_source_try_mlock(up_base, up_len, &ul);
    (void)ds4_gpu_expert_source_try_mlock(down_base, down_len, &dl);
    g_expert_source_admit_ms += ds4_gpu_now_ms() - t0;

    e->used = true;
    e->model_map = model_map;
    e->layer = layer;
    e->expert = expert;
    e->charged_bytes = charge;
    e->gate_base = gate_base;
    e->up_base = up_base;
    e->down_base = down_base;
    e->gate_len = gate_len;
    e->up_len = up_len;
    e->down_len = down_len;
    e->gate_locked = gl;
    e->up_locked = ul;
    e->down_locked = dl;
    ds4_gpu_expert_source_lru_link_head(idx);
    if (layer < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        expert < DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) {
        g_expert_source_index[layer][expert] = idx;
    }
    g_expert_source_used_bytes += charge;
    if (g_expert_source_used_bytes > g_expert_source_peak_bytes) g_expert_source_peak_bytes = g_expert_source_used_bytes;
    g_expert_source_admissions++;
    if (gl || ul || dl) g_expert_source_locked_entries++;
    else g_expert_source_soft_entries++;
}

static void ds4_gpu_expert_source_cache_note(
        const void *model_map,
        uint32_t layer,
        uint32_t n_active,
        const uint32_t *active_ids,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!ds4_gpu_expert_source_is_enabled() || !ds4_gpu_expert_source_layer_allowed(layer) ||
        !active_ids || n_active == 0 || layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) {
        return;
    }
    if (!ds4_gpu_expert_source_init()) return;

    for (uint32_t i = 0; i < n_active; i++) {
        const uint32_t expert = active_ids[i];
        if (expert >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) continue;
        g_expert_source_requests++;
        int32_t idx = ds4_gpu_expert_source_find(model_map, layer, expert);
        if (idx >= 0) {
            g_expert_source_hits++;
            ds4_gpu_expert_source_lru_touch(idx);
            continue;
        }
        g_expert_source_misses++;
        uint32_t seen = ++g_expert_source_seen[layer][expert];
        if (seen >= g_expert_source_admit_after) {
            if (g_expert_source_async_prefetch && !g_expert_source_hard_copy) {
                ds4_gpu_expert_source_prefetch_enqueue(model_map,
                                                       expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes);
            } else {
                ds4_gpu_expert_source_admit(model_map,
                                            layer,
                                            expert,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
            }
        }
    }
    if (g_expert_source_interval != 0 &&
        (g_expert_source_requests % g_expert_source_interval) == 0) {
        ds4_gpu_expert_source_print("live");
    }
}

static const ds4_metal_expert_source_entry *ds4_gpu_expert_source_hard_find_ex(
        const void *model_map,
        uint32_t layer,
        uint32_t expert,
        bool touch_lru) {
    if (!g_expert_source_hard_copy || !g_expert_source_entries) return NULL;
    int32_t idx = ds4_gpu_expert_source_find(model_map, layer, expert);
    if (idx < 0) return NULL;
    ds4_metal_expert_source_entry *e = &g_expert_source_entries[idx];
    if (!e->hard_copy || !e->gate_copy || !e->up_copy || !e->down_copy) return NULL;
    if (touch_lru) ds4_gpu_expert_source_lru_touch(idx);
    return e;
}

static const ds4_metal_expert_source_entry *ds4_gpu_expert_source_hard_find(
        const void *model_map,
        uint32_t layer,
        uint32_t expert) {
    return ds4_gpu_expert_source_hard_find_ex(model_map, layer, expert, true);
}

static uint32_t ds4_gpu_expert_gather_threads(void) {
    static int initialized;
    static uint32_t threads;
    if (!initialized) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_GATHER_THREADS", 1u);
        if (v < 1u) v = 1u;
        if (v > 16u) v = 16u;
        threads = (uint32_t)v;
        initialized = 1;
        if (threads > 1u && ds4_gpu_expert_offload_enabled() && !ds4_gpu_expert_offload_direct_enabled()) {
            fprintf(stderr,
                    "ds4: A3 expert CPU gather parallel copy enabled: %u threads\n",
                    threads);
        }
    }
    return threads;
}

/* ---- project.md P0.1/P1.1: expert IO instrumentation + single-copy pread ----
 *
 * Measured bottleneck (execution log 2026-06-10): the A3 gather moves cold
 * expert bytes at ~1.5GB/s because every byte pays a 16KiB mmap page fault
 * (SSD -> page cache) plus a memcpy (page cache -> Shared scratch), serialized
 * behind a per-layer command drain.  DS4_METAL_EXPERT_PREAD=1 replaces the
 * fault+memcpy double copy with one pread() per expert tensor straight into
 * the scratch MTLBuffer.  DS4_METAL_EXPERT_PREAD_NOCACHE=1 additionally reads
 * through a separate F_NOCACHE descriptor so the ~1.7GiB/token cold stream
 * stops evicting hot page-cache pages (A/B knob, off by default).
 * DS4_METAL_EXPERT_IO_PROFILE=1 prints one ds4-io line per routed-MoE layer
 * call decomposing wall time into fault/memcpy/pread/drain so the account in
 * project.md P0.1 can be settled from a normal speed run. */

static int ds4_gpu_pread_full(int fd, void *dst, uint64_t src_off, size_t len) {
    uint8_t *p = (uint8_t *)dst;
    while (len > 0) {
        ssize_t r = pread(fd, p, len, (off_t)src_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += (size_t)r;
        src_off += (uint64_t)r;
        len -= (size_t)r;
    }
    return 1;
}

/* Resolve once from the serial gather entry (before worker threads spawn) so
 * the cached env lookups never race. */
static int ds4_gpu_expert_pread_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_PREAD") > 0 ? 1 : 0;
        if (cached && (g_model_fd < 0 || g_model_fd_conflict)) {
            fprintf(stderr,
                    "ds4: DS4_METAL_EXPERT_PREAD=1 requested but no unambiguous model fd; "
                    "falling back to mmap gather\n");
            cached = 0;
        }
        if (cached) {
            fprintf(stderr, "ds4: expert gather single-copy pread enabled (fd=%d)\n",
                    g_model_fd);
        }
    }
    return cached;
}

/* Wave 24: verify/prefill batch gathers read hundreds of MiB of cold expert
 * bytes per layer with ~zero reuse (batch hit_mib==0 in every profile run),
 * yet a kc=12 verify round streams ~5.8GiB through the coordinator page
 * cache and evicts the mmap-resident backbone (Q8 attention/shared) pages.
 * The next single-token frame then re-faults the backbone inside the GPU
 * command execution: decode drain_ms ballooned 14.5 -> 46ms/layer (spikes to
 * 855ms on the first layers after a verify), inflating round-1 from ~550ms to
 * ~2.3s.  Route batch-site cold preads through a separate F_NOCACHE
 * descriptor so one-shot batch bytes stop evicting hot pages.  Default on;
 * DS4_METAL_EXPERT_BATCH_NOCACHE=0 restores the shared cached fd (A/B). */
static int g_expert_gather_nocache_call;   /* set on the serial encode path per gather call */

static int ds4_gpu_expert_batch_nocache_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_METAL_EXPERT_BATCH_NOCACHE");
        cached = (v && *v && v[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* Wave 25 A/B verdict: NOCACHE won on first-touch prefill chunks (smoke
 * 2.02 -> 2.09) but lost on kc<=16 verify rounds (code-edit 1.68 -> 1.48):
 * verify unions overlap decode-hot experts and neighbouring rounds, so
 * bypassing the cache re-reads warm bytes from the slow mini SSD.  Keep
 * NOCACHE for big (prefill-sized) batches only; backbone protection against
 * verify eviction moves to the mlock pin (see ds4_gpu_backbone_mlock_register). */
static uint32_t ds4_gpu_expert_batch_nocache_min_tokens(void) {
    static uint32_t cached;
    static int init;
    if (!init) {
        const char *v = getenv("DS4_METAL_EXPERT_BATCH_NOCACHE_MIN");
        uint64_t n = v ? strtoull(v, NULL, 10) : 0;
        /* Wave 35: 24 was calibrated when K=16 capped verify at 17 rows --
         * verify could never trip it.  The K=64 ladder now sends kc=25/33
         * verify batches (the two big rounds, 11.8s of 21.3s r2 total) which
         * 24 misclassifies as prefill-sized cold streams: their unions lose
         * both the warm bytes of the neighbouring round (consecutive copy
         * rounds share most experts) and the decode-hot overlap -- exactly
         * the case the wave-25 verdict measured as a loss (1.68 vs 1.48).
         * 64 keeps every verify batch (protocol max 64 rows) on the cached
         * fd; prefill frames (128 tokens) stay NOCACHE. */
        if (n == 0) n = 64;
        if (n > UINT32_MAX) n = UINT32_MAX;
        cached = (uint32_t)n;
        init = 1;
    }
    return cached;
}

static int ds4_gpu_expert_pread_fd_nocache(void) {
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

static int ds4_gpu_expert_pread_fd(void) {
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

static int ds4_gpu_expert_io_profile_enabled(void) {
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
static uint64_t g_io_prof_fault_ns;
static uint64_t g_io_prof_copy_ns;
static uint64_t g_io_prof_pread_ns;
static uint64_t g_io_prof_remote_ns;
static uint64_t g_io_prof_cold_bytes;
static uint64_t g_io_prof_hit_bytes;
static uint64_t g_io_prof_remote_bytes;
static uint64_t g_io_prof_pread_fallbacks;   /* cumulative, never reset */
static volatile uint64_t g_io_prof_touch_sink;

static void ds4_gpu_expert_io_prof_reset(void) {
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
static uint64_t g_pf_pred_hits, g_pf_pred_total;

static void ds4_gpu_expert_io_prof_report(const char *site,
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
static uint64_t ds4_gpu_expert_touch_pages(const uint8_t *p, size_t len) {
    uint64_t sink = 0;
    const size_t page = 16384;
    for (size_t i = 0; i < len; i += page) sink += p[i];
    if (len) sink += p[len - 1];
    return sink;
}

/* ---- project.md P2.1: cross-layer router prediction + async expert prefetch --
 *
 * Decode is a serial chain: gather(L) -> GPU(L) -> drain -> gather(L+1) ...
 * The router of layer L+1 is a tiny F16 [n_expert][n_embd] matrix, and the
 * residual stream changes slowly across one layer, so evaluating router(L+1)
 * on layer L's MoE input already ranks the true top-6 with high overlap
 * (Pre-Attention Expert Prediction, arXiv:2511.10676).  While the GPU computes
 * layer L we re-evaluate router(L+1) on a background thread and issue
 * F_RDADVISE read-ahead for the predicted experts' gate/up/down ranges, so by
 * the time gather(L+1) preads them they are (partially) page-cache hits.
 * Predictions are purely advisory: a miss only wastes read bandwidth, the
 * gather path stays bit-exact.  DS4_METAL_EXPERT_PREFETCH_AHEAD=1 enables,
 * DS4_METAL_EXPERT_PREFETCH_TOP (default 8) controls the prediction margin. */

#define DS4_METAL_PF_MAX_EMBD 8192u
#define DS4_METAL_PF_MAX_TOP 32u
#define DS4_METAL_PF_QUEUE 4u

typedef struct {
    int valid;
    int gate_inp_is_f32;      /* early layers carry an F32 router matrix */
    const void *model_map;
    uint64_t gate_inp_off;    /* F16/F32 [n_expert][n_embd] rows */
    uint64_t probs_bias_off;  /* F32 [n_expert], UINT64_MAX = absent */
    uint64_t gate_exps_off;
    uint64_t up_exps_off;
    uint64_t down_exps_off;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t n_embd;
    uint32_t n_expert;
    uint64_t hash_off;        /* token-id hash routing LUT (I32 [k][n_vocab]); UINT64_MAX = score routing */
    uint32_t hash_k;
    uint32_t hash_rows;
} ds4_metal_layer_router;

static ds4_metal_layer_router g_layer_router[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

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

static int ds4_gpu_expert_prefetch_top(void) {
    static int cached;
    if (cached == 0) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_PREFETCH_TOP", 8u);
        if (v < 1u) v = 1u;
        if (v > DS4_METAL_PF_MAX_TOP) v = DS4_METAL_PF_MAX_TOP;
        cached = (int)v;
    }
    return cached;
}

/* Predict a RANGE of upcoming layers (L+1 .. L+depth) from the same hidden
 * snapshot.  The paced read-ahead can only use SSD-idle windows (~drain time
 * per layer); depth D gives each layer's prediction up to D windows to finish
 * before its gather arrives.  Closest deadline (L+1) is advised first, and
 * mincore skips ranges an earlier job already pulled in, so deeper lookahead
 * degrades gracefully.  Accuracy cost is measurable via the ds4-io pf= field
 * (predictions for L+d use a hidden state that is d-1 layers stale).
 * Honors DS4_METAL_EXPERT_PREFETCH_DELTA as a legacy alias. */
static uint32_t ds4_gpu_expert_prefetch_depth(void) {
    static uint32_t cached;
    if (cached == 0) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_PREFETCH_DEPTH",
                                     ds4_gpu_env_u64("DS4_METAL_EXPERT_PREFETCH_DELTA", 1u));
        if (v < 1u) v = 1u;
        if (v > 4u) v = 4u;
        cached = (uint32_t)v;
    }
    return cached;
}

/* Wave 36: batch rows.  A kc-token verify batch activates a 50-74 expert
 * union per layer, but the old job carried ONE hidden row -- the predictor
 * covered ~8 of them and batch gathers ran effectively unpredicted (all
 * cold).  Batch jobs snapshot up to this many stride-sampled rows and the
 * predictor unions their per-row top-k (neighbouring tokens share experts,
 * so 16 sampled rows recover most of the union). */
#define DS4_METAL_PF_MAX_ROWS 16u

typedef struct {
    uint32_t layer;                      /* layer being predicted (= L+1) */
    int kind;                            /* 0 = score prediction from x; 1 = token-hash note */
    int token;
    uint32_t n_rows;                     /* 0/1 = single-x decode job; >1 = batch union job */
    float x[DS4_METAL_PF_MAX_EMBD];      /* router-input snapshot from layer L (kind 0) */
    float xrows[DS4_METAL_PF_MAX_ROWS * DS4_METAL_PF_MAX_EMBD]; /* batch row snapshots */
} ds4_metal_pf_job;

/* Scheduling (v2, after the 1.56->1.49 regression): decode keeps the SSD
 * ~100% busy, so read-ahead must never compete with the foreground gather --
 * the first version did, and the extra traffic cost more than the overlap
 * gained.  v2 is "polite": the prefetch thread only issues advisories in
 * 1MiB chunks while no foreground gather is running (g_gather_active), skips
 * ranges that are already page-cache resident (mincore), works in router
 * score order so the scarce idle window is spent on the most likely experts,
 * and a newer prediction supersedes an unfinished older one (latest-wins). */
static pthread_mutex_t g_pf_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pf_cv = PTHREAD_COND_INITIALIZER;
static ds4_metal_pf_job g_pf_slot;             /* single slot, latest wins */
static volatile uint32_t g_pf_seq;             /* bumped per enqueue */
static uint32_t g_pf_picked_seq;               /* last seq picked by the worker */
static int g_pf_shutdown;
static uint64_t g_pf_enqueued, g_pf_superseded, g_pf_advise_fail, g_pf_skip_cached;
static volatile int g_gather_active;           /* foreground gather/stream running */
/* Prediction sets for accuracy accounting: mark[layer][expert] == gen[layer]
 * means "expert was in the latest predicted set for that layer".  Written by
 * the prefetch thread, read by the gather entry; stats-only, races benign. */
static uint32_t g_pf_pred_gen[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];
static uint32_t g_pf_pred_mark[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS][DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS];
/* Current decode token (router hook).  Hash-routed early layers' expert sets
 * are an exact function of this id. */
static volatile int g_pf_token = -1;
static int g_pf_saw_nonhash = 1;

static float ds4_gpu_pf_softplus(float v) {
    if (v > 20.0f) return v;
    if (v < -20.0f) return expf(v);
    return log1pf(expf(v));
}

static void ds4_gpu_expert_prefetch_advise(int fd, uint64_t off, uint64_t len) {
    struct radvisory ra;
    ra.ra_offset = (off_t)off;
    ra.ra_count = (int)len;
    if (fcntl(fd, F_RDADVISE, &ra) != -1) return;
    g_pf_advise_fail++;
    /* Portable fallback: pull the range through the page cache with plain
     * preads into a throwaway buffer (single prefetch thread => static ok). */
    static uint8_t scratch[262144];
    uint64_t done = 0;
    while (done < len) {
        size_t chunk = sizeof(scratch);
        if ((uint64_t)chunk > len - done) chunk = (size_t)(len - done);
        if (!ds4_gpu_pread_full(fd, scratch, off + done, chunk)) break;
        done += chunk;
    }
}

/* Issue read-ahead in small chunks, yielding whenever a foreground gather is
 * running and aborting as soon as a newer prediction supersedes this one.
 * Returns 0 on abort. */
static int ds4_gpu_expert_prefetch_advise_paced(int fd, uint64_t off, uint64_t len, uint32_t my_seq) {
    const uint64_t chunk = 1ull << 20;
    uint64_t done = 0;
    while (done < len) {
        while (g_gather_active) {
            if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
            usleep(200);
        }
        if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
        uint64_t n = chunk;
        if (len - done < n) n = len - done;
        ds4_gpu_expert_prefetch_advise(fd, off + done, n);
        done += n;
    }
    return 1;
}

/* True when (almost) the whole file range is already page-cache resident, in
 * which case read-ahead would only waste the idle window.  Ground truth via
 * mincore on the model mapping (same UBC pages the preads hit). */
static int ds4_gpu_expert_range_mostly_cached(const void *model_map, uint64_t off, uint64_t len) {
    char vec[8192];          /* stack: called from prefetch + remote fetch threads */
    static size_t page;
    if (page == 0) {
        long p = sysconf(_SC_PAGESIZE);
        page = p > 0 ? (size_t)p : 16384u;
    }
    const uintptr_t addr = (uintptr_t)model_map + (uintptr_t)off;
    const uintptr_t start = addr & ~(uintptr_t)(page - 1u);
    const size_t span = (size_t)(addr + len - start);
    const size_t npages = (span + page - 1u) / page;
    if (npages == 0 || npages > sizeof(vec)) return 0;
    if (mincore((void *)start, span, vec) != 0) return 0;
    size_t resident = 0;
    for (size_t i = 0; i < npages; i++) resident += (size_t)(vec[i] & 1);
    return resident * 10u >= npages * 9u;   /* >=90% resident: skip */
}

/* Defined in the staging section below; predict_one dispatches to them. */
static int ds4_gpu_expert_stage_enabled(void);
static void ds4_gpu_expert_stage_arm(uint32_t layer, const int *top_idx, uint32_t n_top, int token);
static void ds4_gpu_expert_stage_arm_if_new(uint32_t layer, const int *top_idx, uint32_t n_top, int token);
static void ds4_gpu_expert_hash_note_run(int token);

/* Predict one layer, then either hand the predicted experts to the remote
 * staging pipeline (stage_this: pulled from the peer's SSD into RAM during
 * this layer's window) or issue local paced read-ahead advisories.  Returns 0
 * when superseded/shutting down. */
static int ds4_gpu_expert_prefetch_predict_one(uint32_t layer, const float *x, uint32_t my_seq,
                                               int stage_this) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return 1;
    const ds4_metal_layer_router *r = &g_layer_router[layer];
    if (!r->valid || g_model_fd < 0) return 1;
    /* Token-id hash routing (early layers): the expert set is an exact
     * function of the current token -- 100% "prediction" accuracy. */
    if (r->hash_off != UINT64_MAX) {
        const int tok = g_pf_token;
        if (tok < 0 || (uint32_t)tok >= r->hash_rows) return 1;
        const int32_t *row = (const int32_t *)((const uint8_t *)r->model_map + r->hash_off) +
                             (size_t)tok * r->hash_k;
        int hidx[16];
        uint32_t m = 0;
        for (uint32_t i = 0; i < r->hash_k && i < 16u; i++) {
            if (row[i] >= 0 && (uint32_t)row[i] < r->n_expert) hidx[(int)m++] = row[i];
        }
        if (m == 0) return 1;
        const uint32_t hgen = ++g_pf_pred_gen[layer];
        for (uint32_t i = 0; i < m; i++) g_pf_pred_mark[layer][(uint32_t)hidx[i]] = hgen;
        if (stage_this && ds4_gpu_expert_stage_enabled()) {
            ds4_gpu_expert_stage_arm_if_new(layer, hidx, m, tok);
            return 1;
        }
        const int hfd = g_model_fd;
        for (uint32_t i = 0; i < m; i++) {
            const uint64_t e = (uint64_t)hidx[i];
            const uint64_t hsoff[3] = {
                r->gate_exps_off + e * r->gate_expert_bytes,
                r->up_exps_off + e * r->gate_expert_bytes,
                r->down_exps_off + e * r->down_expert_bytes,
            };
            const uint64_t hslen[3] = { r->gate_expert_bytes, r->gate_expert_bytes,
                                        r->down_expert_bytes };
            for (uint32_t s = 0; s < 3u; s++) {
                if (ds4_gpu_expert_range_mostly_cached(r->model_map, hsoff[s], hslen[s])) {
                    g_pf_skip_cached++;
                    continue;
                }
                if (!ds4_gpu_expert_prefetch_advise_paced(hfd, hsoff[s], hslen[s], my_seq)) {
                    return 0;
                }
            }
        }
        return 1;
    }
    const uint8_t *map = (const uint8_t *)r->model_map;
    const __fp16 *w16 = (const __fp16 *)(map + r->gate_inp_off);
    const float *w32 = (const float *)(map + r->gate_inp_off);
    const float *bias = r->probs_bias_off != UINT64_MAX ?
        (const float *)(map + r->probs_bias_off) : NULL;
    const uint32_t n_top = (uint32_t)ds4_gpu_expert_prefetch_top();
    /* Opportunistic staging depth: the top-n_top experts saturate the link
     * budget; a couple of extra candidates ride the idle tail of the window
     * (fetched last, in score order) and convert some prediction misses. */
    uint32_t n_top_total = n_top + (uint32_t)ds4_gpu_env_u64("DS4_METAL_EXPERT_STAGE_EXTRA", 2u);
    if (n_top_total > DS4_METAL_PF_MAX_TOP) n_top_total = DS4_METAL_PF_MAX_TOP;

    /* score = sqrt(softplus(gate_inp . x)) + bias, matching the selection rule
     * in layer_topk_selected_experts (ranking only; weights don't matter). */
    int top_idx[DS4_METAL_PF_MAX_TOP];
    float top_score[DS4_METAL_PF_MAX_TOP];
    for (uint32_t i = 0; i < n_top_total; i++) { top_idx[i] = -1; top_score[i] = -1e30f; }
    for (uint32_t e = 0; e < r->n_expert; e++) {
        float acc = 0.0f;
        if (r->gate_inp_is_f32) {
            const float *row = w32 + (size_t)e * r->n_embd;
            for (uint32_t d = 0; d < r->n_embd; d++) acc += row[d] * x[d];
        } else {
            const __fp16 *row = w16 + (size_t)e * r->n_embd;
            for (uint32_t d = 0; d < r->n_embd; d++) acc += (float)row[d] * x[d];
        }
        float s = sqrtf(ds4_gpu_pf_softplus(acc));
        if (bias) s += bias[e];
        if (s <= top_score[n_top_total - 1u]) continue;
        uint32_t j = n_top_total - 1u;
        while (j > 0u && s > top_score[j - 1u]) {
            top_score[j] = top_score[j - 1u];
            top_idx[j] = top_idx[j - 1u];
            j--;
        }
        top_score[j] = s;
        top_idx[j] = (int)e;
    }

    /* Mark the headline predicted set first (prediction-accuracy accounting
     * stays comparable across TOP settings), then stage/advise in descending
     * score order; abort mid-way once a newer prediction lands. */
    const uint32_t gen = ++g_pf_pred_gen[layer];
    for (uint32_t i = 0; i < n_top; i++) {
        if (top_idx[i] < 0) break;
        g_pf_pred_mark[layer][(uint32_t)top_idx[i]] = gen;
    }
    if (stage_this && ds4_gpu_expert_stage_enabled()) {
        ds4_gpu_expert_stage_arm(layer, top_idx, n_top_total, g_pf_token);
        return 1;   /* fetch threads stage from the peer; no local advisories */
    }
    const int fd = g_model_fd;   /* read-ahead must hit the page cache, never the F_NOCACHE fd */
    for (uint32_t i = 0; i < n_top; i++) {
        if (top_idx[i] < 0) break;
        const uint64_t e = (uint64_t)top_idx[i];
        const uint64_t seg_off[3] = {
            r->gate_exps_off + e * r->gate_expert_bytes,
            r->up_exps_off + e * r->gate_expert_bytes,
            r->down_exps_off + e * r->down_expert_bytes,
        };
        const uint64_t seg_len[3] = {
            r->gate_expert_bytes, r->gate_expert_bytes, r->down_expert_bytes,
        };
        for (uint32_t s = 0; s < 3u; s++) {
            if (ds4_gpu_expert_range_mostly_cached(r->model_map, seg_off[s], seg_len[s])) {
                g_pf_skip_cached++;
                continue;
            }
            if (!ds4_gpu_expert_prefetch_advise_paced(fd, seg_off[s], seg_len[s], my_seq)) {
                return 0;   /* superseded or shutting down */
            }
        }
    }
    return 1;
}

/* Wave 36: batch-union prediction.  Per sampled row, score all experts with
 * the same selection rule as predict_one, keep each row's top-k, and union
 * them (per-expert max score orders the advisories).  Local advisories only:
 * a 50-70 expert union (~400MiB) belongs to the local fast path -- the
 * 16-slot peer staging machinery cannot hold it and TB cannot deliver it in
 * one drain window.  The advisories flow while g_gather_active is clear,
 * i.e. exactly inside the next layer's GPU drain (~50-60ms, disk idle), so
 * the following gather's cached-fd preads hit warm pages.  Wrong predictions
 * cost only wasted read-ahead; correctness is untouched. */
static int ds4_gpu_expert_prefetch_predict_union(uint32_t layer,
                                                 const float *rows,
                                                 uint32_t n_rows,
                                                 uint32_t my_seq) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return 1;
    const ds4_metal_layer_router *r = &g_layer_router[layer];
    if (!r->valid || g_model_fd < 0) return 1;
    if (r->hash_off != UINT64_MAX) return 1;  /* hash routing is per token id */
    if (r->n_expert == 0 || r->n_expert > 1024u) return 1;
    const uint8_t *map = (const uint8_t *)r->model_map;
    const __fp16 *w16 = (const __fp16 *)(map + r->gate_inp_off);
    const float *w32 = (const float *)(map + r->gate_inp_off);
    const float *bias = r->probs_bias_off != UINT64_MAX ?
        (const float *)(map + r->probs_bias_off) : NULL;
    const uint32_t row_top = (uint32_t)ds4_gpu_expert_prefetch_top();

    float best[1024];
    for (uint32_t e = 0; e < r->n_expert; e++) best[e] = -1e30f;

    for (uint32_t ri = 0; ri < n_rows; ri++) {
        if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
        const float *x = rows + (size_t)ri * DS4_METAL_PF_MAX_EMBD;
        int top_idx[DS4_METAL_PF_MAX_TOP];
        float top_score[DS4_METAL_PF_MAX_TOP];
        uint32_t k = row_top;
        if (k > DS4_METAL_PF_MAX_TOP) k = DS4_METAL_PF_MAX_TOP;
        for (uint32_t i = 0; i < k; i++) { top_idx[i] = -1; top_score[i] = -1e30f; }
        for (uint32_t e = 0; e < r->n_expert; e++) {
            float acc = 0.0f;
            if (r->gate_inp_is_f32) {
                const float *row = w32 + (size_t)e * r->n_embd;
                for (uint32_t dd = 0; dd < r->n_embd; dd++) acc += row[dd] * x[dd];
            } else {
                const __fp16 *row = w16 + (size_t)e * r->n_embd;
                for (uint32_t dd = 0; dd < r->n_embd; dd++) acc += (float)row[dd] * x[dd];
            }
            float s = sqrtf(ds4_gpu_pf_softplus(acc));
            if (bias) s += bias[e];
            if (s <= top_score[k - 1u]) continue;
            uint32_t j = k - 1u;
            while (j > 0u && s > top_score[j - 1u]) {
                top_score[j] = top_score[j - 1u];
                top_idx[j] = top_idx[j - 1u];
                j--;
            }
            top_score[j] = s;
            top_idx[j] = (int)e;
        }
        for (uint32_t i = 0; i < k; i++) {
            if (top_idx[i] < 0) break;
            if (top_score[i] > best[(uint32_t)top_idx[i]]) {
                best[(uint32_t)top_idx[i]] = top_score[i];
            }
        }
    }

    /* Advise the union in descending score order: the idle window is spent on
     * the most confident experts first. */
    const int fd = g_model_fd;
    for (;;) {
        if (g_pf_seq != my_seq || g_pf_shutdown) return 0;
        float bs = -1e29f;
        int be = -1;
        for (uint32_t e = 0; e < r->n_expert; e++) {
            if (best[e] > bs) { bs = best[e]; be = (int)e; }
        }
        if (be < 0 || bs <= -1e29f) break;
        best[be] = -1e30f;
        const uint64_t eo = (uint64_t)be;
        const uint64_t seg_off[3] = {
            r->gate_exps_off + eo * r->gate_expert_bytes,
            r->up_exps_off + eo * r->gate_expert_bytes,
            r->down_exps_off + eo * r->down_expert_bytes,
        };
        const uint64_t seg_len[3] = { r->gate_expert_bytes, r->gate_expert_bytes,
                                      r->down_expert_bytes };
        for (uint32_t s = 0; s < 3u; s++) {
            if (ds4_gpu_expert_range_mostly_cached(r->model_map, seg_off[s], seg_len[s])) {
                g_pf_skip_cached++;
                continue;
            }
            if (!ds4_gpu_expert_prefetch_advise_paced(fd, seg_off[s], seg_len[s], my_seq)) {
                return 0;
            }
        }
    }
    return 1;
}

/* job->layer is the first predicted layer (L+1); predict L+1..L+depth from
 * the same hidden snapshot, closest deadline first.  Only d==0 is staged:
 * its slot is never re-armed while its layer's gather can still read it.
 * kind==1 jobs carry only a token id and arm the exact hash-routed layers
 * (moved off the graph thread: arming spins on the slot's busy gate). */
static void ds4_gpu_expert_prefetch_run_job(const ds4_metal_pf_job *job, uint32_t my_seq) {
    if (job->kind == 1) {
        ds4_gpu_expert_hash_note_run(job->token);
        return;
    }
    if (job->n_rows > 1u) {
        /* Batch union job: depth 1 only (deeper layers from the same snapshot
         * degrade, and the window barely fits one union). */
        (void)ds4_gpu_expert_prefetch_predict_union(job->layer, job->xrows,
                                                    job->n_rows, my_seq);
        return;
    }
    const uint32_t depth = ds4_gpu_expert_prefetch_depth();
    for (uint32_t d = 0; d < depth; d++) {
        if (g_pf_seq != my_seq || g_pf_shutdown) return;
        if (!ds4_gpu_expert_prefetch_predict_one(job->layer + d, job->x, my_seq, d == 0u)) return;
    }
}

static void *ds4_gpu_expert_prefetch_thread(void *arg) {
    (void)arg;
    static ds4_metal_pf_job local;
    uint32_t my_seq = 0;
    for (;;) {
        pthread_mutex_lock(&g_pf_mu);
        while (!g_pf_shutdown && g_pf_seq == my_seq) {
            pthread_cond_wait(&g_pf_cv, &g_pf_mu);
        }
        if (g_pf_shutdown) {
            pthread_mutex_unlock(&g_pf_mu);
            break;
        }
        my_seq = g_pf_seq;
        g_pf_picked_seq = my_seq;
        memcpy(&local, &g_pf_slot, sizeof(local));
        pthread_mutex_unlock(&g_pf_mu);
        ds4_gpu_expert_prefetch_run_job(&local, my_seq);
    }
    return NULL;
}

/* Called from the serial graph thread only (no init race). */
static int ds4_gpu_expert_prefetch_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_PREFETCH_AHEAD") > 0 ? 1 : 0;
        if (cached && g_model_fd < 0) {
            fprintf(stderr,
                    "ds4: DS4_METAL_EXPERT_PREFETCH_AHEAD=1 requested but no model fd; disabled\n");
            cached = 0;
        }
        if (cached) {
            pthread_t th;
            if (pthread_create(&th, NULL, ds4_gpu_expert_prefetch_thread, NULL) == 0) {
                fprintf(stderr,
                        "ds4: cross-layer expert prefetch enabled (top %d predicted experts "
                        "read ahead per next layer)\n",
                        ds4_gpu_expert_prefetch_top());
            } else {
                cached = 0;
            }
        }
    }
    return cached;
}

static void ds4_gpu_expert_prefetch_enqueue(uint32_t next_layer, const float *x, uint32_t n_embd) {
    if (next_layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const ds4_metal_layer_router *r = &g_layer_router[next_layer];
    if (!r->valid || r->n_embd != n_embd) return;
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_seq != g_pf_picked_seq) g_pf_superseded++;   /* old job replaced unstarted/aborted */
    g_pf_slot.layer = next_layer;
    g_pf_slot.kind = 0;
    g_pf_slot.token = g_pf_token;
    g_pf_slot.n_rows = 1;
    memcpy(g_pf_slot.x, x, (size_t)n_embd * sizeof(float));
    g_pf_seq++;
    g_pf_enqueued++;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mu);
}

/* Wave 36: batch snapshot.  x is the row-major [n_tokens x n_embd] FFN input
 * of a verify batch; stride-sample up to DS4_METAL_PF_MAX_ROWS rows so the
 * union prediction sees the whole token range.  Row sampling keeps the
 * per-job score cost ~16 router matvecs (~15-25ms on the prefetch thread,
 * inside the gather+drain window). */
static void ds4_gpu_expert_prefetch_enqueue_batch(uint32_t next_layer,
                                                  const float *x,
                                                  uint32_t n_embd,
                                                  uint32_t n_tokens) {
    if (next_layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    if (!x || n_tokens < 2u) return;
    const ds4_metal_layer_router *r = &g_layer_router[next_layer];
    if (!r->valid || r->n_embd != n_embd) return;
    uint32_t n_rows = n_tokens;
    if (n_rows > DS4_METAL_PF_MAX_ROWS) n_rows = DS4_METAL_PF_MAX_ROWS;
    uint32_t stride = n_tokens / n_rows;
    if (stride == 0) stride = 1;
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_seq != g_pf_picked_seq) g_pf_superseded++;
    g_pf_slot.layer = next_layer;
    g_pf_slot.kind = 0;
    g_pf_slot.token = g_pf_token;
    uint32_t m = 0;
    for (uint32_t t = 0; t < n_tokens && m < n_rows; t += stride, m++) {
        memcpy(g_pf_slot.xrows + (size_t)m * DS4_METAL_PF_MAX_EMBD,
               x + (size_t)t * n_embd,
               (size_t)n_embd * sizeof(float));
    }
    g_pf_slot.n_rows = m;
    g_pf_seq++;
    g_pf_enqueued++;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mu);
}

/* Non-blocking token note from the graph thread: the exact hash-layer arming
 * spins on slot busy gates, so it runs on the prediction thread instead. */
static void ds4_gpu_expert_prefetch_enqueue_hash(int token) {
    pthread_mutex_lock(&g_pf_mu);
    if (g_pf_seq != g_pf_picked_seq) g_pf_superseded++;
    g_pf_slot.layer = 0;
    g_pf_slot.kind = 1;
    g_pf_slot.token = token;
    g_pf_seq++;
    g_pf_enqueued++;
    pthread_cond_signal(&g_pf_cv);
    pthread_mutex_unlock(&g_pf_mu);
}

/* ---- project.md P2.1 x P2.2: prediction-driven remote expert staging ----
 *
 * Racing the per-layer gather cursor against the local pread threads cannot
 * use the Thunderbolt link well: a decode layer exposes only 18 units for
 * ~10ms and 8 local threads claim them instantly, while a remote unit costs a
 * ~2ms round trip that then sits on the layer's critical path (measured:
 * rfetch stuck at ~5MiB/layer).  Staging inverts it: the router prediction
 * for layer L+1 (≈80% top-6 coverage) dispatches the predicted experts to the
 * remote fetch threads which pull them into a RAM staging slot DURING layer
 * L's entire gather+GPU window (~14ms x 4.5GB/s >> 54MiB, no tail
 * constraint).  At layer L+1's gather, staged experts are RAM memcpys; only
 * mispredicted experts touch the local SSD.  Two slots alternate by layer
 * parity; only the first predicted layer (d==0) is staged, so a slot is never
 * re-armed while the gather of its layer can still read it.  Stage bytes are
 * advisory: a missing/partial entry just falls back to the normal local read. */

#define DS4_METAL_STAGE_MAX_EXPERTS 16u

/* Shared with the remote-fetch worker section below (defined here so the
 * prediction thread can wake the fetch threads after arming a slot). */
typedef struct ds4_metal_expert_gather_ctx ds4_metal_expert_gather_ctx;
static ds4_metal_expert_gather_ctx * volatile g_rf_ctx;
static pthread_mutex_t g_rf_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_rf_cv = PTHREAD_COND_INITIALIZER;
static volatile int g_rf_link_down;

typedef struct {
    volatile uint32_t gen;                 /* bumped on (re)arm; 0 = never armed */
    volatile uint32_t busy;                /* fetchers inside this slot (re-arm gate) */
    uint32_t layer;
    int token;                             /* decode token this arm belongs to */
    uint32_t n;
    volatile uint32_t next;                /* atomic claim index for fetch threads */
    uint32_t ids[DS4_METAL_STAGE_MAX_EXPERTS];
    volatile uint32_t ready_gen[DS4_METAL_STAGE_MAX_EXPERTS];
    uint64_t gate_off, up_off, down_off;
    uint64_t gate_eb, down_eb;
    uint64_t stride;                       /* per-expert bytes in buf */
    uint8_t *buf;
    size_t buf_cap;
} ds4_metal_stage_slot;

static ds4_metal_stage_slot g_stage[2];
static volatile int g_stage_recent;   /* parity of the most recently armed slot */
static uint64_t g_stage_hit_bytes_total, g_stage_armed, g_stage_fetched, g_stage_dropped;
static uint64_t g_stage_slots_total;  /* sum of armed expert counts (completion denominator) */

static int ds4_gpu_expert_stage_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_STAGE") > 0 ? 1 : 0;
        /* Wave 33: staging needs working fetch connections to the peer, not
         * specifically the forward-dial client.  Accept mode (worker side:
         * peer dials in, DS4_DIST_EXPERT_FETCH_ACCEPT_PORT) provides the same
         * connections -- and the measured asymmetry was exactly this gate:
         * coordinator decode layers hit 50-100% staged RAM (hit_mib 20-40 of
         * 40.5) while worker decode ran 100% cold (hit_mib=0.0 all run). */
        if (cached && !getenv("DS4_DIST_EXPERT_FETCH_HOST") &&
            !getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT")) cached = 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: predicted experts staged from peer SSD into RAM one layer ahead\n");
        }
    }
    return cached;
}

/* Arm a staging slot for a freshly predicted layer.  Callable from the
 * prediction thread (score routing) and the graph thread (token-hash routing
 * hook), so the whole re-arm is serialized by a mutex. */
static pthread_mutex_t g_stage_arm_mu = PTHREAD_MUTEX_INITIALIZER;

static void ds4_gpu_expert_stage_arm(uint32_t layer,
                                     const int *top_idx,
                                     uint32_t n_top,
                                     int token) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const ds4_metal_layer_router *r = &g_layer_router[layer];
    if (!r->valid) return;
    pthread_mutex_lock(&g_stage_arm_mu);
    ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    const uint64_t stride = 2u * r->gate_expert_bytes + r->down_expert_bytes;
    const size_t need = (size_t)stride * DS4_METAL_STAGE_MAX_EXPERTS;
    if (!s->buf || s->buf_cap < need) {
        free(s->buf);
        s->buf = (uint8_t *)malloc(need);
        s->buf_cap = s->buf ? need : 0;
        if (!s->buf) {
            pthread_mutex_unlock(&g_stage_arm_mu);
            return;
        }
    }
    s->gen++;                       /* invalidates all ready flags */
    __sync_synchronize();
    /* Wait out fetchers still writing this slot's buffer under the old gen:
     * a new-gen fetcher must never interleave writes with an in-flight recv
     * draining into the same region.  Off the gather critical path;
     * in-flight units finish in ~2ms. */
    while (s->busy) usleep(50);
    s->layer = layer;
    s->token = token;
    s->gate_off = r->gate_exps_off;
    s->up_off = r->up_exps_off;
    s->down_off = r->down_exps_off;
    s->gate_eb = r->gate_expert_bytes;
    s->down_eb = r->down_expert_bytes;
    s->stride = stride;
    uint32_t n = n_top;
    if (n > DS4_METAL_STAGE_MAX_EXPERTS) n = DS4_METAL_STAGE_MAX_EXPERTS;
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (top_idx[i] < 0) break;
        s->ids[m++] = (uint32_t)top_idx[i];
    }
    s->n = m;
    __sync_synchronize();
    s->next = 0;
    g_stage_recent = (int)(layer & 1u);
    g_stage_armed++;
    g_stage_slots_total += m;
    /* Wake the remote fetch threads parked on the shared condvar. */
    pthread_mutex_lock(&g_rf_mu);
    pthread_cond_broadcast(&g_rf_cv);
    pthread_mutex_unlock(&g_rf_mu);
    if ((g_stage_armed & 63u) == 0) {
        fprintf(stderr,
                "ds4-stage: armed=%llu fetched=%llu dropped=%llu hit_gib=%.2f "
                "(completion %.0f%%)\n",
                (unsigned long long)g_stage_armed,
                (unsigned long long)g_stage_fetched,
                (unsigned long long)g_stage_dropped,
                (double)g_stage_hit_bytes_total / 1073741824.0,
                g_stage_slots_total ? 100.0 * (double)g_stage_fetched /
                    (double)g_stage_slots_total : 0.0);
    }
    pthread_mutex_unlock(&g_stage_arm_mu);
}

/* Skip the re-arm when the slot already holds this (layer, token): the
 * token-hash hook and the prediction chain may both try to arm hash layers. */
static void ds4_gpu_expert_stage_arm_if_new(uint32_t layer,
                                            const int *top_idx,
                                            uint32_t n_top,
                                            int token) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    if (s->gen != 0 && s->layer == layer && s->token == token) return;
    ds4_gpu_expert_stage_arm(layer, top_idx, n_top, token);
}

/* Prediction-thread side of the token note: arm exact staging for the
 * earliest hash layer of each slot parity; the prediction chain (also exact
 * for hash layers) covers the rest. */
static void ds4_gpu_expert_hash_note_run(int token) {
    int armed_parity[2] = { 0, 0 };
    for (uint32_t l = 0; l < 8u && l < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS; l++) {
        const ds4_metal_layer_router *r = &g_layer_router[l];
        if (!r->valid || r->hash_off == UINT64_MAX) continue;
        const int p = (int)(l & 1u);
        if (armed_parity[p]) continue;
        armed_parity[p] = 1;
        if (token < 0 || (uint32_t)token >= r->hash_rows) continue;
        const int32_t *row = (const int32_t *)((const uint8_t *)r->model_map + r->hash_off) +
                             (size_t)token * r->hash_k;
        int hidx[16];
        uint32_t m = 0;
        for (uint32_t i = 0; i < r->hash_k && i < 16u; i++) {
            if (row[i] >= 0 && (uint32_t)row[i] < r->n_expert) hidx[m++] = row[i];
        }
        if (m) {
            const uint32_t hgen = ++g_pf_pred_gen[l];
            for (uint32_t i = 0; i < m; i++) g_pf_pred_mark[l][(uint32_t)hidx[i]] = hgen;
            ds4_gpu_expert_stage_arm_if_new(l, hidx, m, token);
        }
        if (armed_parity[0] && armed_parity[1]) break;
    }
}

/* Graph-thread hook from ds4_gpu_router_select_tensor: the first hash-mode
 * router select of each decode forward reveals the token id, which exactly
 * determines the hash-routed layers' experts.  MUST NOT BLOCK: arming spins
 * on slot busy gates, so it is queued to the prediction thread (latest-wins
 * is correct here -- at a forward boundary any pending older job is stale). */
static void ds4_gpu_expert_router_note(int token, int hash_mode) {
    if (!hash_mode) {
        g_pf_saw_nonhash = 1;
        return;
    }
    if (!g_pf_saw_nonhash) {
        g_pf_token = token;
        return;   /* later hash layer of the same forward */
    }
    g_pf_saw_nonhash = 0;
    g_pf_token = token;
    if (!ds4_gpu_expert_stage_enabled()) return;
    ds4_gpu_expert_prefetch_enqueue_hash(token);
}

/* Fetch-thread side: claim and stage one expert of slot s.  Returns 0 when
 * the slot has no more work.  The three tensor segments are pipelined on the
 * connection (send all, then recv all) so the server pread, the wire
 * transfer and the request RTT overlap: ~2.5ms per expert instead of ~5
 * (measured gap between 79.6% prediction accuracy and only 54% staged hits
 * was fetch completion, not prediction). */
static int ds4_gpu_expert_stage_fetch_one(ds4_metal_stage_slot *s, int conn_slot) {
    const uint32_t gen = s->gen;
    const uint32_t i = __sync_fetch_and_add(&s->next, 1u);
    if (i >= s->n) return 0;
    __sync_fetch_and_add(&s->busy, 1u);
    if (s->gen != gen) {
        __sync_fetch_and_sub(&s->busy, 1u);
        return 0;
    }
    const uint64_t e = (uint64_t)s->ids[i];
    uint8_t *dst = s->buf + (size_t)i * s->stride;
    const uint64_t seg_off[3] = {
        s->gate_off + e * s->gate_eb,
        s->up_off + e * s->gate_eb,
        s->down_off + e * s->down_eb,
    };
    const uint64_t seg_len[3] = { s->gate_eb, s->gate_eb, s->down_eb };
    uint64_t dst_off = 0;
    int ok = 1;

    /* Locally cached experts never touch the link. */
    int all_cached = g_model_map_ptr != NULL;
    for (uint32_t k = 0; k < 3u && all_cached; k++) {
        all_cached = ds4_gpu_expert_range_mostly_cached(g_model_map_ptr, seg_off[k], seg_len[k]);
    }
    if (all_cached || g_rf_link_down) {
        for (uint32_t k = 0; k < 3u && ok; k++) {
            ok = ds4_gpu_pread_full(g_model_fd, dst + dst_off, seg_off[k], (size_t)seg_len[k]);
            dst_off += seg_len[k];
        }
    } else {
        /* Pipeline: send all three requests, then drain the responses. */
        uint32_t sent = 0;
        while (sent < 3u && ds4_dist_expert_fetch_send(conn_slot, seg_off[sent],
                                                       (uint32_t)seg_len[sent])) {
            sent++;
        }
        for (uint32_t k = 0; k < 3u; k++) {
            if (k < sent) {
                /* Must drain in order even if superseded: the connection's
                 * response stream stays aligned, and a stale write to this
                 * region is safe because re-arming waits on s->busy. */
                if (!ds4_dist_expert_fetch_recv(conn_slot, dst + dst_off, (uint32_t)seg_len[k])) {
                    if (!g_rf_link_down) {
                        g_rf_link_down = 1;
                        fprintf(stderr,
                                "ds4: expert remote fetch link down; gather continues local-only\n");
                    }
                    ok = ds4_gpu_pread_full(g_model_fd, dst + dst_off, seg_off[k],
                                            (size_t)seg_len[k]) && ok;
                }
            } else {
                ok = ds4_gpu_pread_full(g_model_fd, dst + dst_off, seg_off[k],
                                        (size_t)seg_len[k]) && ok;
            }
            dst_off += seg_len[k];
        }
    }
    if (ok && s->gen == gen) {
        __sync_synchronize();
        s->ready_gen[i] = gen;
        __sync_fetch_and_add(&g_stage_fetched, 1u);
    } else if (!ok || s->gen != gen) {
        __sync_fetch_and_add(&g_stage_dropped, 1u);
    }
    __sync_fetch_and_sub(&s->busy, 1u);
    return 1;
}

static int ds4_gpu_expert_stage_has_work(void) {
    for (int k = 0; k < 2; k++) {
        if (g_stage[k].gen != 0 && g_stage[k].next < g_stage[k].n) return 1;
    }
    return 0;
}

/* Is (layer, expert) part of the armed prediction set (ready or not)?  Used
 * by the gather to decide whether a short wait for in-flight staged bytes
 * beats an immediate slow local read. */
static int ds4_gpu_expert_stage_pending(uint32_t layer, uint32_t id) {
    const ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    if (s->gen == 0 || s->layer != layer) return 0;
    const uint32_t n = s->n;
    for (uint32_t i = 0; i < n && i < DS4_METAL_STAGE_MAX_EXPERTS; i++) {
        if (s->ids[i] == id) return 1;
    }
    return 0;
}

/* A staged-but-late expert finishes within ~2.5ms (pipelined fetch); local
 * cold pread costs ~2.7ms+ of the slow disk *and* its bandwidth.  Waiting a
 * bounded moment for in-flight bytes converts late completions into hits.
 * 0 disables. */
static uint32_t ds4_gpu_expert_stage_wait_us(void) {
    static uint32_t cached = UINT32_MAX;
    if (cached == UINT32_MAX) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_STAGE_WAIT_US", 2500u);
        if (v > 20000u) v = 20000u;
        cached = (uint32_t)v;
    }
    return cached;
}

/* Gather side: return the staged bytes for (layer, expert, part) or NULL.
 * Safe by construction: the slot for layer L is only re-armed by the
 * prediction made during layer L+1's gather, after L's gather finished. */
static const uint8_t *ds4_gpu_expert_stage_find(uint32_t layer,
                                                uint32_t id,
                                                uint32_t part,
                                                uint64_t *len_out) {
    const ds4_metal_stage_slot *s = &g_stage[layer & 1u];
    const uint32_t gen = s->gen;
    if (gen == 0 || s->layer != layer) return NULL;
    for (uint32_t i = 0; i < s->n; i++) {
        if (s->ids[i] != id) continue;
        if (s->ready_gen[i] != gen) return NULL;
        const uint8_t *base = s->buf + (size_t)i * s->stride;
        switch (part) {
        case 0: *len_out = s->gate_eb; return base;
        case 1: *len_out = s->gate_eb; return base + s->gate_eb;
        default: *len_out = s->down_eb; return base + 2u * s->gate_eb;
        }
    }
    return NULL;
}

/* Accuracy accounting: compare this layer's actual active set against the
 * latest prediction for it (stats only; harmless races). */
static void ds4_gpu_expert_prefetch_note_actual(uint32_t layer, const uint32_t *ids, uint32_t n) {
    if (layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS) return;
    const uint32_t gen = g_pf_pred_gen[layer];
    if (gen == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] >= DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS) continue;
        g_pf_pred_total++;
        if (g_pf_pred_mark[layer][ids[i]] == gen) g_pf_pred_hits++;
    }
}

struct ds4_metal_expert_gather_ctx {   /* typedef'd forward at the staging slots */
    const void *model_map;
    const uint8_t *map;
    uint8_t *gate_dst;
    uint8_t *up_dst;
    uint8_t *down_dst;
    const uint32_t *active_ids;
    uint32_t n_active;
    uint32_t n_expert_total;
    uint32_t layer_index;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t next_slot;          /* atomic unit cursor (unit = slot*3 + part) */
    uint32_t done;               /* atomic completed-unit counter */
    int use_pread;
    int pread_fd;
    int io_profile;
    int ok;
};

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t done_cv;
    pthread_t threads[16];
    uint32_t n_threads;
    uint32_t target_workers;
    uint32_t active_workers;
    uint64_t generation;
    uint64_t completed_generation;
    int shutdown;
    ds4_metal_expert_gather_ctx *ctx;
} ds4_metal_expert_gather_pool;

static ds4_metal_expert_gather_pool g_expert_gather_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
};

/* One gather work unit = one tensor (gate/up/down) of one active expert.
 * Per-tensor units triple the available IO queue depth versus per-expert
 * units: a 6-expert decode layer exposes 18 concurrent preads instead of 6,
 * which matters because random NVMe bandwidth scales with QD. */
/* Resolve one gather unit to (source file offset, destination, length) plus
 * an optional in-RAM hard-copy source.  Shared by the local copy path and the
 * remote-fetch workers.  Returns 0 on a bad expert id. */
static int ds4_gpu_expert_gather_unit_resolve(
        ds4_metal_expert_gather_ctx *ctx,
        uint32_t unit,
        uint64_t *src_off,
        uint8_t **dst,
        uint64_t *len,
        const uint8_t **hard_src) {
    const uint32_t slot = unit / 3u;
    const uint32_t part = unit % 3u;   /* 0=gate 1=up 2=down */
    const uint32_t id = ctx->active_ids[slot];
    if (id >= ctx->n_expert_total) return 0;
    const ds4_metal_expert_source_entry *src_entry =
        ds4_gpu_expert_source_hard_find_ex(ctx->model_map, ctx->layer_index, id, false);
    *hard_src = NULL;
    switch (part) {
    case 0:
        *len = ctx->gate_expert_bytes;
        *src_off = ctx->gate_offset + (uint64_t)id * ctx->gate_expert_bytes;
        *dst = ctx->gate_dst + (uint64_t)slot * ctx->gate_expert_bytes;
        if (src_entry) *hard_src = (const uint8_t *)src_entry->gate_copy;
        break;
    case 1:
        *len = ctx->gate_expert_bytes;
        *src_off = ctx->up_offset + (uint64_t)id * ctx->gate_expert_bytes;
        *dst = ctx->up_dst + (uint64_t)slot * ctx->gate_expert_bytes;
        if (src_entry) *hard_src = (const uint8_t *)src_entry->up_copy;
        break;
    default:
        *len = ctx->down_expert_bytes;
        *src_off = ctx->down_offset + (uint64_t)id * ctx->down_expert_bytes;
        *dst = ctx->down_dst + (uint64_t)slot * ctx->down_expert_bytes;
        if (src_entry) *hard_src = (const uint8_t *)src_entry->down_copy;
        break;
    }
    return 1;
}

static void ds4_gpu_expert_gather_copy_unit(ds4_metal_expert_gather_ctx *ctx, uint32_t unit) {
    uint64_t len = 0;
    uint64_t src_off = 0;
    uint8_t *dst = NULL;
    const uint8_t *hard_src = NULL;
    if (!ds4_gpu_expert_gather_unit_resolve(ctx, unit, &src_off, &dst, &len, &hard_src)) {
        ctx->ok = 0;
        return;
    }
    /* Staged one layer ahead from the peer's SSD?  RAM copy beats any disk. */
    if (!hard_src) {
        const uint32_t expert_id = ctx->active_ids[unit / 3u];
        uint64_t slen = 0;
        const uint8_t *staged = ds4_gpu_expert_stage_find(ctx->layer_index,
                                                          expert_id,
                                                          unit % 3u,
                                                          &slen);
        if (!staged) {
            /* Predicted but still in flight: a bounded wait for RAM bytes
             * beats an immediate cold read of the slow local disk. */
            const uint32_t wait_us = ds4_gpu_expert_stage_wait_us();
            if (wait_us > 0 &&
                ds4_gpu_expert_stage_pending(ctx->layer_index, expert_id)) {
                const double deadline = ds4_gpu_now_ms() + (double)wait_us / 1000.0;
                do {
                    usleep(100);
                    staged = ds4_gpu_expert_stage_find(ctx->layer_index, expert_id,
                                                       unit % 3u, &slen);
                } while (!staged && ds4_gpu_now_ms() < deadline);
            }
        }
        if (staged && slen == len) {
            const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
            memcpy(dst, staged, (size_t)len);
            if (ctx->io_profile) {
                __sync_fetch_and_add(&g_io_prof_copy_ns,
                                     (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
                __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
            }
            __sync_fetch_and_add(&g_stage_hit_bytes_total, len);
            return;
        }
    }
    if (hard_src) {
        const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
        memcpy(dst, hard_src, (size_t)len);
        if (ctx->io_profile) {
            __sync_fetch_and_add(&g_io_prof_copy_ns,
                                 (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
            __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
        }
        return;
    }
    /* P1.1 single-copy path: SSD -> Shared scratch in one pread, no page
     * fault, no second memcpy.  Bytes are identical to the mmap copy (same
     * file offsets), so logits are bit-exact by construction. */
    if (ctx->use_pread) {
        const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
        if (ds4_gpu_pread_full(ctx->pread_fd, dst, src_off, (size_t)len)) {
            if (ctx->io_profile) {
                __sync_fetch_and_add(&g_io_prof_pread_ns,
                                     (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
                __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
            }
            return;
        }
        /* pread failed (EIO/short read): fall back to the proven mmap copy. */
        __sync_fetch_and_add(&g_io_prof_pread_fallbacks, 1u);
    }
    if (ctx->io_profile) {
        const double tf0 = ds4_gpu_now_ms();
        g_io_prof_touch_sink += ds4_gpu_expert_touch_pages(ctx->map + src_off, (size_t)len);
        const double tc0 = ds4_gpu_now_ms();
        __sync_fetch_and_add(&g_io_prof_fault_ns, (uint64_t)((tc0 - tf0) * 1e6));
        memcpy(dst, ctx->map + src_off, (size_t)len);
        __sync_fetch_and_add(&g_io_prof_copy_ns,
                             (uint64_t)((ds4_gpu_now_ms() - tc0) * 1e6));
        __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
        return;
    }
    memcpy(dst, ctx->map + src_off, (size_t)len);
}

/* ---- project.md P2.2 low-cost variant: remote expert fetch workers ----
 *
 * P0.3 measured the coordinator's SSD at ~2.5GB/s (every pattern) and the
 * worker's at 5.5-6.7GB/s -- and the worker's disk idles during the
 * coordinator's pipeline half.  Both machines map the byte-identical GGUF, so
 * these workers pull gather units off the same atomic cursor as the local
 * pread threads but satisfy them via ds4_dist_expert_fetch() from the peer's
 * disk over Thunderbolt: aggregate supply = local SSD + TB link.  Failure of
 * the link permanently degrades to local-only (correctness never depends on
 * the peer).  Enabled by DS4_DIST_EXPERT_FETCH_HOST on the puller side and
 * DS4_DIST_EXPERT_FETCH_SERVE=1 on the serving side.
 * (g_rf_ctx/g_rf_mu/g_rf_cv/g_rf_link_down live next to the staging slots
 * above so the prediction thread can wake the fetch threads.) */

typedef struct {
    uint64_t src_off;
    uint8_t *dst;
    uint64_t len;
    double t0;
} ds4_metal_rf_pending;

/* Complete a claimed unit locally (used for page-cache hits, tail units and
 * link-failure fallbacks so correctness never depends on the peer).  Prefers
 * the single-copy pread path like the local gather workers; cached pages make
 * pread a RAM copy anyway. */
static void ds4_gpu_expert_remote_local_finish(ds4_metal_expert_gather_ctx *ctx,
                                               uint64_t src_off,
                                               uint8_t *dst,
                                               uint64_t len) {
    const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
    if (!(ctx->use_pread &&
          ds4_gpu_pread_full(ctx->pread_fd, dst, src_off, (size_t)len))) {
        memcpy(dst, ctx->map + src_off, (size_t)len);
    }
    if (ctx->io_profile) {
        __sync_fetch_and_add(&g_io_prof_copy_ns,
                             (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
        __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
    }
}

/* Last few gather units must stay local: a ~2ms remote round-trip claimed at
 * the end of a layer stalls every finished local thread (measured: the tail
 * ate the mid-layer remote gains).  Reserve about one unit per local gather
 * thread; locals retire those in a single parallel round. */
static uint32_t ds4_gpu_expert_remote_tail_reserve(void) {
    static uint32_t cached;
    static int initialized;
    if (!initialized) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_REMOTE_TAIL_RESERVE",
                                     ds4_gpu_expert_gather_threads());
        if (v > 64u) v = 64u;
        cached = (uint32_t)v;
        initialized = 1;
    }
    return cached;
}

/* The Thunderbolt link idles for the whole worker pipeline half (~250ms per
 * decoded token); TCP slow-start-after-idle then collapses the congestion
 * window, so the first staged fetches of the next token (the hash layers 0/1,
 * tightest deadlines!) crawl through the ramp-up.  While parked, each fetch
 * thread keeps its connection warm with a small periodic read.  0 disables. */
static uint32_t ds4_gpu_expert_link_keepalive_ms(void) {
    static uint32_t cached = UINT32_MAX;
    if (cached == UINT32_MAX) {
        /* Default off: measured a net regression (2.03 -> 1.79 t/s) with 40ms
         * keepalives; kept as an A/B knob only. */
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_LINK_KEEPALIVE_MS", 0u);
        if (v > 1000u) v = 1000u;
        cached = (uint32_t)v;
    }
    return cached;
}

static void *ds4_gpu_expert_remote_fetch_worker(void *arg) {
    const int slot = (int)(intptr_t)arg;
    uint8_t *ka_buf = NULL;
    const uint32_t ka_bytes = 262144u;
    for (;;) {
        int do_keepalive = 0;
        pthread_mutex_lock(&g_rf_mu);
        while (!g_rf_ctx && !ds4_gpu_expert_stage_has_work()) {
            const uint32_t ka_ms = ds4_gpu_expert_link_keepalive_ms();
            if (ka_ms == 0 || g_rf_link_down) {
                pthread_cond_wait(&g_rf_cv, &g_rf_mu);
                continue;
            }
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += (long)ka_ms * 1000000L;
            while (ts.tv_nsec >= 1000000000L) { ts.tv_nsec -= 1000000000L; ts.tv_sec++; }
            if (pthread_cond_timedwait(&g_rf_cv, &g_rf_mu, &ts) == ETIMEDOUT &&
                !g_rf_ctx && !ds4_gpu_expert_stage_has_work()) {
                do_keepalive = 1;
                break;
            }
        }
        ds4_metal_expert_gather_ctx *ctx = (ds4_metal_expert_gather_ctx *)g_rf_ctx;
        pthread_mutex_unlock(&g_rf_mu);

        if (do_keepalive) {
            if (!ka_buf) ka_buf = (uint8_t *)malloc(ka_bytes);
            if (ka_buf && !g_rf_link_down) {
                (void)ds4_dist_expert_fetch(slot, 0, ka_buf, ka_bytes);
            }
            continue;
        }

        if (!ctx) {
            /* Staging work: pull the predicted next-layer experts from the
             * peer into the RAM slots (runs during the current layer's local
             * gather + GPU window; no layer-tail constraint).  Most recently
             * armed slot first: its deadline is closest and link time spent
             * on the stale parity is wasted. */
            const int first = g_stage_recent & 1;
            for (int pass = 0; pass < 2; pass++) {
                ds4_metal_stage_slot *s = &g_stage[(first + pass) & 1];
                while (s->gen != 0 && ds4_gpu_expert_stage_fetch_one(s, slot)) {}
            }
            continue;
        }

        /* Pipelined fetch: keep up to 2 requests in flight on this slot's
         * connection so the server-side pread and the wire transfer overlap
         * and the request RTT leaves the per-unit critical path.  Units whose
         * bytes are already in the LOCAL page cache (prefetch read-ahead or
         * natural reuse) are finished with a RAM copy instead of spending
         * Thunderbolt bandwidth on them. */
        ds4_metal_rf_pending pend[2];
        uint32_t npend = 0;
        int exhausted = 0;
        const uint32_t total_units = ctx->n_active * 3u;
        const uint32_t tail_reserve = ds4_gpu_expert_remote_tail_reserve();
        const uint32_t remote_cutoff =
            total_units > tail_reserve ? total_units - tail_reserve : 0u;
        for (;;) {
            while (!exhausted && npend < 2u) {
                if (g_rf_link_down) { exhausted = 1; break; }
                /* Tail guard: never claim into the reserved zone (peek, then
                 * re-check after the claim in case of a race). */
                if (ctx->next_slot >= remote_cutoff) { exhausted = 1; break; }
                const uint32_t unit = __sync_fetch_and_add(&ctx->next_slot, 1u);
                if (unit >= total_units) { exhausted = 1; break; }
                uint64_t len = 0;
                uint64_t src_off = 0;
                uint8_t *dst = NULL;
                const uint8_t *hard_src = NULL;
                if (!ds4_gpu_expert_gather_unit_resolve(ctx, unit, &src_off, &dst, &len, &hard_src)) {
                    ctx->ok = 0;
                    __sync_fetch_and_add(&ctx->done, 1u);
                    continue;
                }
                if (hard_src) {
                    const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
                    memcpy(dst, hard_src, (size_t)len);
                    if (ctx->io_profile) {
                        __sync_fetch_and_add(&g_io_prof_copy_ns,
                                             (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
                        __sync_fetch_and_add(&g_io_prof_hit_bytes, len);
                    }
                    __sync_fetch_and_add(&ctx->done, 1u);
                    continue;
                }
                if (unit >= remote_cutoff) {
                    /* Raced into the tail zone: keep it local. */
                    ds4_gpu_expert_remote_local_finish(ctx, src_off, dst, len);
                    __sync_fetch_and_add(&ctx->done, 1u);
                    exhausted = 1;
                    break;
                }
                if (ds4_gpu_expert_range_mostly_cached(ctx->map, src_off, len)) {
                    ds4_gpu_expert_remote_local_finish(ctx, src_off, dst, len);
                    __sync_fetch_and_add(&ctx->done, 1u);
                    continue;
                }
                if (!ds4_dist_expert_fetch_send(slot, src_off, (uint32_t)len)) {
                    if (!g_rf_link_down) {
                        g_rf_link_down = 1;
                        fprintf(stderr,
                                "ds4: expert remote fetch link down; gather continues local-only\n");
                    }
                    ds4_gpu_expert_remote_local_finish(ctx, src_off, dst, len);
                    __sync_fetch_and_add(&ctx->done, 1u);
                    exhausted = 1;
                    break;
                }
                pend[npend].src_off = src_off;
                pend[npend].dst = dst;
                pend[npend].len = len;
                pend[npend].t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
                npend++;
            }
            if (npend == 0) break;
            /* Drain the oldest in-flight response. */
            if (ds4_dist_expert_fetch_recv(slot, pend[0].dst, (uint32_t)pend[0].len)) {
                if (ctx->io_profile) {
                    __sync_fetch_and_add(&g_io_prof_remote_ns,
                                         (uint64_t)((ds4_gpu_now_ms() - pend[0].t0) * 1e6));
                    __sync_fetch_and_add(&g_io_prof_remote_bytes, pend[0].len);
                }
            } else {
                if (!g_rf_link_down) {
                    g_rf_link_down = 1;
                    fprintf(stderr,
                            "ds4: expert remote fetch link down; gather continues local-only\n");
                }
                ds4_gpu_expert_remote_local_finish(ctx, pend[0].src_off, pend[0].dst, pend[0].len);
            }
            __sync_fetch_and_add(&ctx->done, 1u);
            pend[0] = pend[1];
            npend--;
        }

        /* Wait for the entry to retire this ctx so we don't spin on an
         * exhausted cursor (pointer compare only; never dereferenced here). */
        pthread_mutex_lock(&g_rf_mu);
        while (g_rf_ctx == ctx) pthread_cond_wait(&g_rf_cv, &g_rf_mu);
        pthread_mutex_unlock(&g_rf_mu);
    }
    return NULL;
}

/* Lazy init on the serial gather path: by the first gather both engines are
 * up (the distributed session is already established), so the connect is
 * race-free with the peer's listener.
 * Wave 27/28: the first connect can hit a transient EHOSTUNREACH (Thunderbolt
 * bridge ARP starves under prefill rfetch saturation -- observed on the
 * worker->coordinator reverse-efetch dial while ping/route were fine once the
 * link went quiet).  A one-shot failure used to disable remote fetch for the
 * whole run; wave-27's 8x2s retries all landed inside the ~150s code-edit
 * prefill storm and still died.  Retry every 2s for up to 150 attempts
 * (~5min) so attempts reach the decode phase where the link is idle. */
static pthread_mutex_t g_rf_init_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_rf_live;            /* live fetch worker threads */

static int ds4_gpu_expert_remote_fetch_slots(void) {
    static int attempts;
    static double retry_after_ms;
    if (g_rf_live > 0) return g_rf_live;
    /* Wave 30 accept mode (reverse-established transport): the kick thread
     * owns the whole init -- the gather path must not dial anything. */
    if (getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT")) return g_rf_live;
    const char *host = getenv("DS4_DIST_EXPERT_FETCH_HOST");
    if (!host || !host[0]) return 0;
    /* Wave 29: also dialed from the quiet-window kick thread; serialize with
     * the gather path (a contended caller just reports "not yet"). */
    if (pthread_mutex_trylock(&g_rf_init_mu) != 0) return 0;
    if (g_rf_live > 0 || attempts >= 150) {
        const int n = g_rf_live;
        pthread_mutex_unlock(&g_rf_init_mu);
        return n;
    }
    const double now = ds4_gpu_now_ms();
    if (attempts > 0 && now < retry_after_ms) {
        pthread_mutex_unlock(&g_rf_init_mu);
        return 0;
    }
    attempts++;
    retry_after_ms = now + 2000.0;
    const int port = (int)ds4_gpu_env_u64("DS4_DIST_EXPERT_FETCH_PORT", 5606u);
    int conns = (int)ds4_gpu_env_u64("DS4_DIST_EXPERT_FETCH_CONNS", 3u);
    if (conns > 8) conns = 8;
    const int n = ds4_dist_expert_fetch_client_init(host, port, conns, g_model_map_size);
    for (int i = 0; i < n; i++) {
        pthread_t th;
        if (pthread_create(&th, NULL, ds4_gpu_expert_remote_fetch_worker,
                           (void *)(intptr_t)i) == 0) {
            pthread_detach(th);
            g_rf_live++;
        }
    }
    if (g_rf_live) {
        fprintf(stderr,
                "ds4: expert gather pulls from peer SSD via %d remote fetch thread(s)%s "
                "(attempt %d)\n",
                g_rf_live, attempts > 1 ? " after retry" : "", attempts);
    } else if (attempts >= 150) {
        fprintf(stderr,
                "ds4: expert-fetch disabled after 150 failed connect attempts\n");
    } else if (attempts == 1 || attempts % 16 == 0) {
        fprintf(stderr,
                "ds4: expert-fetch connect attempt %d/150 failed; retrying every 2s\n",
                attempts);
    }
    const int live = g_rf_live;
    pthread_mutex_unlock(&g_rf_init_mu);
    return live;
}

/* Wave 29: the in-run retries almost never see a quiet bridge -- coordinator
 * staging keeps 6 efetch connections saturating the Thunderbolt link for the
 * entire run (prefill AND decode), so the worker's reverse-efetch ARP probes
 * starve from the first gather to the last (observed: 31 straight
 * EHOSTUNREACH).  The only reliably quiet window is right after the worker
 * accepts the coordinator's control connection, before the first prefill
 * frame.  Dial from a short-lived background thread in that window. */
static void *ds4_gpu_expert_remote_fetch_kick_main(void *arg) {
    (void)arg;
    /* Wave 30 accept mode: the worker cannot dial out at all (in-process
     * EHOSTUNREACH every attempt while shell tools succeed), so it LISTENS
     * and the coordinator's serve-dial thread connects in.  This kick fires
     * right after the worker accepts the control connection -- the same
     * moment the coordinator's engine (already up, it just dialed us) starts
     * its serve-dial backoff loop, so the rendezvous is race-free. */
    const char *aport = getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT");
    if (aport && aport[0]) {
        pthread_mutex_lock(&g_rf_init_mu);
        if (g_rf_live == 0) {
            const int port = atoi(aport);
            int conns = (int)ds4_gpu_env_u64("DS4_DIST_EXPERT_FETCH_CONNS", 3u);
            if (conns > 8) conns = 8;
            const int n = ds4_dist_expert_fetch_accept_init(port, conns, g_model_map_size);
            for (int i = 0; i < n; i++) {
                pthread_t th;
                if (pthread_create(&th, NULL, ds4_gpu_expert_remote_fetch_worker,
                                   (void *)(intptr_t)i) == 0) {
                    pthread_detach(th);
                    g_rf_live++;
                }
            }
            if (g_rf_live) {
                fprintf(stderr,
                        "ds4: expert gather pulls from peer SSD via %d remote fetch "
                        "thread(s) (accept mode)\n",
                        g_rf_live);
            }
        }
        pthread_mutex_unlock(&g_rf_init_mu);
        return NULL;
    }
    for (int i = 0; i < 10 && ds4_gpu_expert_remote_fetch_slots() == 0; i++) {
        usleep(700 * 1000);
    }
    return NULL;
}

void ds4_gpu_expert_remote_fetch_kick(void) {
    const char *host = getenv("DS4_DIST_EXPERT_FETCH_HOST");
    const char *aport = getenv("DS4_DIST_EXPERT_FETCH_ACCEPT_PORT");
    if ((!host || !host[0]) && (!aport || !aport[0])) return;
    pthread_t th;
    if (pthread_create(&th, NULL, ds4_gpu_expert_remote_fetch_kick_main, NULL) == 0) {
        pthread_detach(th);
    }
}

typedef struct {
    ds4_metal_expert_gather_pool *pool;
    uint32_t index;
} ds4_metal_expert_gather_worker_arg;

static ds4_metal_expert_gather_worker_arg g_expert_gather_worker_args[16];

static void *ds4_gpu_expert_gather_pool_worker(void *arg) {
    ds4_metal_expert_gather_worker_arg *warg = (ds4_metal_expert_gather_worker_arg *)arg;
    ds4_metal_expert_gather_pool *pool = warg->pool;
    const uint32_t worker_index = warg->index;
    uint64_t seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mu);
        while (!pool->shutdown && pool->generation == seen_generation) {
            pthread_cond_wait(&pool->cv, &pool->mu);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mu);
            break;
        }
        seen_generation = pool->generation;
        const uint32_t participates = worker_index < pool->target_workers;
        ds4_metal_expert_gather_ctx *ctx = participates ? pool->ctx : NULL;
        pthread_mutex_unlock(&pool->mu);

        if (!participates) continue;
        for (;;) {
            const uint32_t unit = __sync_fetch_and_add(&ctx->next_slot, 1u);
            if (unit >= ctx->n_active * 3u) break;
            ds4_gpu_expert_gather_copy_unit(ctx, unit);
            __sync_fetch_and_add(&ctx->done, 1u);
        }

        pthread_mutex_lock(&pool->mu);
        if (pool->active_workers > 0) pool->active_workers--;
        if (pool->active_workers == 0) {
            pool->completed_generation = seen_generation;
            pthread_cond_signal(&pool->done_cv);
        }
        pthread_mutex_unlock(&pool->mu);
    }
    return NULL;
}

static void *ds4_gpu_expert_gather_temp_worker(void *arg) {
    ds4_metal_expert_gather_ctx *ctx = (ds4_metal_expert_gather_ctx *)arg;
    for (;;) {
        const uint32_t unit = __sync_fetch_and_add(&ctx->next_slot, 1u);
        if (unit >= ctx->n_active * 3u) break;
        ds4_gpu_expert_gather_copy_unit(ctx, unit);
        __sync_fetch_and_add(&ctx->done, 1u);
    }
    return NULL;
}

static int ds4_gpu_expert_gather_pool_run(ds4_metal_expert_gather_ctx *ctx, uint32_t nth) {
    if (nth == 0 || nth > 16u) return 0;
    pthread_t th[16];
    int created = 0;
    for (uint32_t i = 0; i < nth; i++) {
        if (pthread_create(&th[i], NULL, ds4_gpu_expert_gather_temp_worker, ctx) == 0) {
            created++;
        }
    }
    for (int i = 0; i < created; i++) {
        (void)pthread_join(th[i], NULL);
    }
    return created == (int)nth ? ctx->ok : 0;
}

static int ds4_gpu_load_layer_experts_to_scratch(
        const void *model_map,
        uint32_t    layer_index,
        uint32_t    n_active,
        const uint32_t *active_ids,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (!model_map || n_expert_total == 0 || !active_ids || n_active == 0) return 0;
    const uint64_t gate_total = (uint64_t)n_active * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)n_active * down_expert_bytes;
    if (gate_total > NSUIntegerMax || down_total > NSUIntegerMax) return 0;

    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_gate,
                                       &g_moe_scratch_gate_bytes,
                                       (NSUInteger)gate_total,
                                       "ds4_moe_scratch_gate") ||
        !ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_up,
                                       &g_moe_scratch_up_bytes,
                                       (NSUInteger)gate_total,
                                       "ds4_moe_scratch_up") ||
        !ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_down,
                                       &g_moe_scratch_down_bytes,
                                       (NSUInteger)down_total,
                                       "ds4_moe_scratch_down")) {
        return 0;
    }

    const uint8_t *map = (const uint8_t *)model_map;
    uint8_t *gate_dst = (uint8_t *)g_moe_scratch_gate.contents;
    uint8_t *up_dst = (uint8_t *)g_moe_scratch_up.contents;
    uint8_t *down_dst = (uint8_t *)g_moe_scratch_down.contents;
    if (!gate_dst || !up_dst || !down_dst) return 0;

    /* Resolve cached env switches here, on the serial entry path, so the
     * lazily-initialized statics never race with gather worker threads. */
    const int io_profile = ds4_gpu_expert_io_profile_enabled();
    int use_pread = ds4_gpu_expert_pread_enabled();
    int pread_fd = -1;
    if (use_pread) {
        if (g_expert_gather_nocache_call) pread_fd = ds4_gpu_expert_pread_fd_nocache();
        if (pread_fd < 0) pread_fd = ds4_gpu_expert_pread_fd();
    }
    if (use_pread && pread_fd < 0) use_pread = 0;
    const uint32_t gather_threads = ds4_gpu_expert_gather_threads();
    const uint32_t total_units = n_active * 3u;
    /* Cursor racing pays on big prefill batches (hundreds of units) and on
     * decode layers the staging could not cover (unregistered routers, layer
     * 0 without a predecessor): their gather window leaves the link idle
     * anyway, and the tail guard keeps remote off the layer's critical end.
     * Staged decode layers skip racing: 8 local threads claim the 18-unit
     * cursor instantly and a ~2ms remote round trip only adds tail. */
    /* NOTE: do NOT gate this on staging readiness.  Letting a low-readiness
     * layer fall back to cursor racing steals the fetch connections from the
     * NEXT layer's staging, which then enters ITS gather low on readiness --
     * the fallback cascades down all layers and converts the whole lookahead
     * pipeline back into in-layer racing (measured: 2.03 -> 1.80 t/s).
     * Armed == staged; the in-flight waits harvest what arrives late. */
    const int layer_is_staged =
        ds4_gpu_expert_stage_enabled() &&
        layer_index < DS4_METAL_EXPERT_PROFILE_MAX_LAYERS &&
        g_stage[layer_index & 1u].gen != 0 &&
        g_stage[layer_index & 1u].layer == layer_index;
    /* Wave 32: units floor.  The old `|| !stage_enabled` arm made EVERY
     * gather race on the worker once accept-mode rfetch went live (worker
     * has no FETCH_HOST => staging off => arm always true).  Worker decode
     * gathers (18 units) finish locally in 2-6ms; a TB round trip to the
     * busy coordinator disk is 6ms+ of pure tail (measured: 426/1104 decode
     * gathers raced, smoke 2.13 -> 2.08 the moment accept mode connected).
     * Keep racing for batch-sized work (>= 96 units: prefill + verify) and
     * for the coordinator's unstaged-decode-layer path; small unstaged
     * gathers stay local. */
    const int remote_on =
        ds4_gpu_expert_remote_fetch_slots() > 0 && !g_rf_link_down &&
        (total_units >= 96u ||
         (ds4_gpu_expert_stage_enabled() && !layer_is_staged));
    g_gather_active = 1;   /* prefetch read-ahead yields while we own the SSD */

    ds4_metal_expert_gather_ctx ctx = {
        .model_map = model_map,
        .map = map,
        .gate_dst = gate_dst,
        .up_dst = up_dst,
        .down_dst = down_dst,
        .active_ids = active_ids,
        .n_active = n_active,
        .n_expert_total = n_expert_total,
        .layer_index = layer_index,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .next_slot = 0,
        .done = 0,
        .use_pread = use_pread,
        .pread_fd = pread_fd,
        .io_profile = io_profile,
        .ok = 1,
    };

    /* Publish the shared cursor so the remote fetch workers can pull units
     * from the peer's SSD in parallel with the local pread threads. */
    if (remote_on) {
        pthread_mutex_lock(&g_rf_mu);
        g_rf_ctx = &ctx;
        pthread_cond_broadcast(&g_rf_cv);
        pthread_mutex_unlock(&g_rf_mu);
    }

    int pool_ok = 0;
    uint32_t nth = gather_threads;
    if (nth > total_units) nth = total_units;
    if (nth > 1u) pool_ok = ds4_gpu_expert_gather_pool_run(&ctx, nth);
    if (!pool_ok) {
        /* Thread pool unavailable/failed: the calling thread drains the same
         * shared cursor inline (remote workers may still help). */
        (void)ds4_gpu_expert_gather_temp_worker(&ctx);
    }

    if (remote_on) {
        /* Local workers are done; remote workers may still have claimed units
         * in flight.  Wait for completion, then retire the ctx (the workers
         * only compare the pointer, never dereference it after retirement). */
        while (ctx.done < total_units) usleep(100);
        pthread_mutex_lock(&g_rf_mu);
        g_rf_ctx = NULL;
        pthread_cond_broadcast(&g_rf_cv);
        pthread_mutex_unlock(&g_rf_mu);
    }
    g_gather_active = 0;
    return ctx.ok;
}

/* ---- project.md P1.2: prefill full-layer sequential streaming ----
 *
 * A prefill chunk of >=128 tokens activates nearly all 256 experts of every
 * routed layer, so the per-expert gather degenerates into ~256 scattered
 * multi-MiB reads.  When the active set covers at least
 * DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT (default 60) percent of the layer,
 * DS4_METAL_EXPERT_FULL_LAYER_STREAM=1 streams the whole gate/up/down expert
 * tensors into scratch as three long sequential reads (chunked across the
 * gather threads) and keeps the original expert ids — no slot remap, the MoE
 * kernels index scratch exactly like the resident direct path. */

static int ds4_gpu_expert_stream_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_FULL_LAYER_STREAM") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: prefill full-layer expert streaming enabled "
                    "(threshold %llu%%, chunk %lluMiB)\n",
                    (unsigned long long)ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT", 60u),
                    (unsigned long long)ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_CHUNK_MB", 16u));
        }
    }
    return cached;
}

static uint32_t ds4_gpu_expert_stream_threshold_pct(void) {
    static uint32_t cached;
    static int initialized;
    if (!initialized) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_THRESHOLD_PCT", 60u);
        if (v < 1u) v = 1u;
        if (v > 100u) v = 100u;
        cached = (uint32_t)v;
        initialized = 1;
    }
    return cached;
}

/* Wave 31: full-layer streaming reads the whole 1728MiB tensor set from the
 * LOCAL disk only -- the stream path has no remote-fetch racing.  Measured on
 * the 49-token verify batch: mini L0-2 streamed at 1.9GB/s = ~950ms/layer
 * while the neighboring raced gathers did ~620MiB in ~130ms at 5-6.8GB/s
 * aggregated over both SSDs.  Whenever racing is live, dense activation is
 * exactly when racing pays the most (>=345 units), so skip streaming and let
 * the raced gather take it.  DS4_METAL_EXPERT_STREAM_RFETCH_BYPASS=0 restores
 * the old always-stream behavior. */
static int ds4_gpu_expert_stream_rfetch_bypass(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_METAL_EXPERT_STREAM_RFETCH_BYPASS");
        cached = (env && env[0] == '0' && env[1] == '\0') ? 0 : 1;
    }
    if (!cached) return 0;
    return ds4_gpu_expert_remote_fetch_slots() > 0 && !g_rf_link_down;
}

static uint64_t ds4_gpu_expert_stream_chunk_bytes(void) {
    static uint64_t cached;
    if (cached == 0) {
        uint64_t mb = ds4_gpu_env_u64("DS4_METAL_EXPERT_STREAM_CHUNK_MB", 16u);
        if (mb < 1u) mb = 1u;
        if (mb > 256u) mb = 256u;
        cached = mb << 20;
    }
    return cached;
}

typedef struct {
    const uint8_t *map;
    int use_pread;
    int pread_fd;
    int io_profile;
    uint64_t seg_src[3];
    uint8_t *seg_dst[3];
    uint64_t seg_len[3];
    uint64_t seg_chunks[3];
    uint64_t chunk_bytes;
    uint64_t total_chunks;
    uint64_t next_chunk;   /* atomic cursor shared by the stream workers */
    int ok;
} ds4_metal_expert_stream_ctx;

static void ds4_gpu_expert_stream_copy_chunk(ds4_metal_expert_stream_ctx *ctx, uint64_t chunk) {
    uint32_t seg = 0;
    uint64_t base = 0;
    while (seg < 3u && chunk >= base + ctx->seg_chunks[seg]) {
        base += ctx->seg_chunks[seg];
        seg++;
    }
    if (seg >= 3u) {
        ctx->ok = 0;
        return;
    }
    const uint64_t off = (chunk - base) * ctx->chunk_bytes;
    uint64_t len = ctx->seg_len[seg] - off;
    if (len > ctx->chunk_bytes) len = ctx->chunk_bytes;
    const double t0 = ctx->io_profile ? ds4_gpu_now_ms() : 0.0;
    if (ctx->use_pread &&
        ds4_gpu_pread_full(ctx->pread_fd,
                           ctx->seg_dst[seg] + off,
                           ctx->seg_src[seg] + off,
                           (size_t)len)) {
        if (ctx->io_profile) {
            __sync_fetch_and_add(&g_io_prof_pread_ns,
                                 (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
            __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
        }
        return;
    }
    if (ctx->use_pread) __sync_fetch_and_add(&g_io_prof_pread_fallbacks, 1u);
    memcpy(ctx->seg_dst[seg] + off, ctx->map + ctx->seg_src[seg] + off, (size_t)len);
    if (ctx->io_profile) {
        __sync_fetch_and_add(&g_io_prof_copy_ns,
                             (uint64_t)((ds4_gpu_now_ms() - t0) * 1e6));
        __sync_fetch_and_add(&g_io_prof_cold_bytes, len);
    }
}

static void *ds4_gpu_expert_stream_worker(void *arg) {
    ds4_metal_expert_stream_ctx *ctx = (ds4_metal_expert_stream_ctx *)arg;
    for (;;) {
        const uint64_t chunk = __sync_fetch_and_add(&ctx->next_chunk, 1ull);
        if (chunk >= ctx->total_chunks) break;
        ds4_gpu_expert_stream_copy_chunk(ctx, chunk);
        if (!ctx->ok) break;
    }
    return NULL;
}

static int ds4_gpu_stream_layer_experts_to_scratch(
        const void *model_map,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        uint32_t    n_expert_total) {
    if (!model_map || n_expert_total == 0) return 0;
    const uint64_t gate_total = (uint64_t)n_expert_total * gate_expert_bytes;
    const uint64_t down_total = (uint64_t)n_expert_total * down_expert_bytes;
    if (gate_total == 0 || down_total == 0 ||
        gate_total > NSUIntegerMax || down_total > NSUIntegerMax) {
        return 0;
    }
    /* Same scratch pool as the per-expert gather: with ~all experts active the
     * gather already sizes scratch to the full layer, so streaming does not
     * raise the peak footprint. */
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_gate,
                                       &g_moe_scratch_gate_bytes,
                                       (NSUInteger)gate_total,
                                       "ds4_moe_scratch_gate") ||
        !ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_up,
                                       &g_moe_scratch_up_bytes,
                                       (NSUInteger)gate_total,
                                       "ds4_moe_scratch_up") ||
        !ds4_gpu_ensure_scratch_buffer(&g_moe_scratch_down,
                                       &g_moe_scratch_down_bytes,
                                       (NSUInteger)down_total,
                                       "ds4_moe_scratch_down")) {
        return 0;
    }
    uint8_t *gate_dst = (uint8_t *)g_moe_scratch_gate.contents;
    uint8_t *up_dst = (uint8_t *)g_moe_scratch_up.contents;
    uint8_t *down_dst = (uint8_t *)g_moe_scratch_down.contents;
    if (!gate_dst || !up_dst || !down_dst) return 0;

    const int io_profile = ds4_gpu_expert_io_profile_enabled();
    int use_pread = ds4_gpu_expert_pread_enabled();
    const int pread_fd = use_pread ? ds4_gpu_expert_pread_fd() : -1;
    if (use_pread && pread_fd < 0) use_pread = 0;

    ds4_metal_expert_stream_ctx ctx = {
        .map = (const uint8_t *)model_map,
        .use_pread = use_pread,
        .pread_fd = pread_fd,
        .io_profile = io_profile,
        .seg_src = { gate_offset, up_offset, down_offset },
        .seg_dst = { gate_dst, up_dst, down_dst },
        .seg_len = { gate_total, gate_total, down_total },
        .chunk_bytes = ds4_gpu_expert_stream_chunk_bytes(),
        .next_chunk = 0,
        .ok = 1,
    };
    for (uint32_t s = 0; s < 3u; s++) {
        ctx.seg_chunks[s] = (ctx.seg_len[s] + ctx.chunk_bytes - 1u) / ctx.chunk_bytes;
        ctx.total_chunks += ctx.seg_chunks[s];
    }

    uint32_t nth = ds4_gpu_expert_gather_threads();
    if ((uint64_t)nth > ctx.total_chunks) nth = (uint32_t)ctx.total_chunks;
    pthread_t th[16];
    uint32_t created = 0;
    g_gather_active = 1;   /* prefetch read-ahead yields while we own the SSD */
    if (nth > 1u) {
        for (uint32_t i = 0; i + 1u < nth && i < 16u; i++) {
            if (pthread_create(&th[created], NULL, ds4_gpu_expert_stream_worker, &ctx) == 0) {
                created++;
            }
        }
    }
    /* The calling thread always participates; the shared cursor guarantees
     * every chunk is copied exactly once even if thread creation failed. */
    (void)ds4_gpu_expert_stream_worker(&ctx);
    for (uint32_t i = 0; i < created; i++) (void)pthread_join(th[i], NULL);
    g_gather_active = 0;
    return ctx.ok;
}

/* Pass 1 of the selected-expert compaction: collect the unique active expert
 * ids of this layer call (first-appearance order) without touching the
 * selected-id buffer, so the caller can pick gather vs full-layer streaming
 * before committing to a slot remap. */
static int ds4_gpu_collect_active_experts(
        id<MTLBuffer> selectedbuf,
        NSUInteger    selected_off,
        uint32_t      n_picks,
        uint32_t      n_expert_total,
        uint32_t     *active_ids,
        uint32_t      active_cap,
        uint32_t     *n_active_out) {
    if (!selectedbuf || !active_ids || !n_active_out || active_cap == 0) return 0;
    if (selectedbuf.storageMode != MTLStorageModeShared) {
        fprintf(stderr,
                "ds4: A3 expert offload requires Shared-storage selected buffer (got mode %lu)\n",
                (unsigned long)selectedbuf.storageMode);
        return 0;
    }
    if (n_expert_total == 0 || n_expert_total > active_cap || active_cap > 1024u) return 0;
    const int32_t *sel_cpu =
        (const int32_t *)((const uint8_t *)selectedbuf.contents + (size_t)selected_off);
    int16_t seen_lut[1024];
    for (uint32_t i = 0; i < active_cap; i++) seen_lut[i] = -1;
    uint32_t n_active = 0;
    for (uint32_t i = 0; i < n_picks; i++) {
        int32_t raw = sel_cpu[i];
        uint32_t id = (raw >= 0 && (uint32_t)raw < n_expert_total) ? (uint32_t)raw : 0u;
        if (seen_lut[id] < 0) {
            if (n_active >= active_cap) return 0;
            seen_lut[id] = 0;
            active_ids[n_active++] = id;
        }
    }
    *n_active_out = n_active;
    return n_active != 0;
}

static int ds4_gpu_cmp_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a;
    const uint32_t y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static int ds4_gpu_expert_sort_ids_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_EXPERT_SORT_IDS") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: expert gather slots sorted by expert id (sequential cold reads)\n");
        }
    }
    return cached;
}

/* Pass 2: optionally sort the active ids ascending (= ascending file offsets,
 * so cold reads walk the GGUF forward instead of jumping around) and rewrite
 * the selected-id buffer from original expert ids to compact scratch slots.
 * Bit-exact either way: per-pick results and their summation order are
 * unchanged, only the scratch slot numbering moves. */
static int ds4_gpu_remap_selected_to_slots(
        id<MTLBuffer> selectedbuf,
        NSUInteger    selected_off,
        uint32_t      n_picks,
        uint32_t      n_expert_total,
        uint32_t     *active_ids,
        uint32_t      n_active) {
    if (!selectedbuf || !active_ids || n_active == 0 || n_active > 1024u) return 0;
    if (ds4_gpu_expert_sort_ids_enabled() && n_active > 1u) {
        qsort(active_ids, n_active, sizeof(uint32_t), ds4_gpu_cmp_u32);
    }
    int16_t compact_lut[1024];
    for (uint32_t i = 0; i < 1024u; i++) compact_lut[i] = -1;
    for (uint32_t s = 0; s < n_active; s++) {
        if (active_ids[s] >= 1024u) return 0;
        compact_lut[active_ids[s]] = (int16_t)s;
    }
    int32_t *sel_cpu =
        (int32_t *)((uint8_t *)selectedbuf.contents + (size_t)selected_off);
    for (uint32_t i = 0; i < n_picks; i++) {
        int32_t raw = sel_cpu[i];
        uint32_t id = (raw >= 0 && (uint32_t)raw < n_expert_total) ? (uint32_t)raw : 0u;
        if (id >= 1024u || compact_lut[id] < 0) return 0;
        sel_cpu[i] = (int32_t)compact_lut[id];
    }
    return 1;
}

static int ds4_gpu_compact_selected_experts(
        id<MTLBuffer> selectedbuf,
        NSUInteger    selected_off,
        uint32_t      n_picks,
        uint32_t      n_expert_total,
        uint32_t     *active_ids,
        uint32_t      active_cap,
        uint32_t     *n_active_out) {
    if (!ds4_gpu_collect_active_experts(selectedbuf, selected_off, n_picks,
                                        n_expert_total, active_ids, active_cap,
                                        n_active_out)) {
        return 0;
    }
    return ds4_gpu_remap_selected_to_slots(selectedbuf, selected_off, n_picks,
                                           n_expert_total, active_ids,
                                           *n_active_out);
}

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        g_wrap_mlock_suppress = 1;   /* routed expert tensors: gathered, never wired */
        id<MTLBuffer> gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
        id<MTLBuffer> up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
        id<MTLBuffer> down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
        g_wrap_mlock_suppress = 0;
        if (!gate_buf || !up_buf || !down_buf) return 0;
        uint32_t source_n_total_expert = n_total_expert;

        /* DS4_METAL_EXPERT_OFFLOAD is meant for the full q2 target model, whose
         * routed experts are deliberately non-resident and must be gathered into
         * compact scratch/pool before the MoE kernels index them.  The MTP support
         * model is mapped as a fully resident model range; forcing its Q4_K routed
         * experts through the A3 scratch path just copies resident bytes again and
         * made distributed MTP a net slowdown.  Leave non-q2 routed tensors on the
         * direct resident-buffer path. */
        const bool a3_expert_offload =
            ds4_gpu_expert_offload_enabled() &&
            !ds4_gpu_expert_offload_direct_enabled() &&
            gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
            down_type == DS4_METAL_TENSOR_Q2_K;
        if (a3_expert_offload) {
            const int io_profile = ds4_gpu_expert_io_profile_enabled();
            g_expert_gather_nocache_call = 0;   /* decode reads stay page-cache friendly */
            const int was_batched = (g_batch_cb != nil);
            double drain_ms = 0.0;
            if (was_batched) {
                const double drain_t0 = io_profile ? ds4_gpu_now_ms() : 0.0;
                if (ds4_gpu_expert_drain_commands("routed MoE drain") == 0) return 0;
                if (io_profile) drain_ms = ds4_gpu_now_ms() - drain_t0;
            }
            /* P2.1: snapshot this layer's router input and kick next-layer
             * router prediction + expert read-ahead on the background thread;
             * it overlaps both this layer's gather and the GPU compute.  The
             * batch was just drained, so x is CPU-visible and final. */
            if (ds4_gpu_expert_prefetch_enabled() &&
                xbuf.storageMode == MTLStorageModeShared) {
                const float *x_cpu = (const float *)((const uint8_t *)xbuf.contents +
                                                     (size_t)ds4_gpu_tensor_offset(x));
                ds4_gpu_expert_prefetch_enqueue(layer_index + 1u, x_cpu, expert_in_dim);
            }
            uint32_t active_ids[1024];
            uint32_t n_active = 0;
            int compact_ok = ds4_gpu_compact_selected_experts(selectedbuf,
                                                              selected_off,
                                                              n_expert,
                                                              n_total_expert,
                                                              active_ids,
                                                              1024,
                                                              &n_active);
            /* Compare this layer's actual active set against the latest
             * prediction made for it (stats feed the ds4-io pf= field). */
            if (compact_ok) {
                ds4_gpu_expert_prefetch_note_actual(layer_index, active_ids, n_active);
            }
            /* Real expert pool is currently limited to the full q2 routed layout.
             * MTP/q4 routed tensors can have different slot sizes and fall back to
             * the proven A3 scratch path. */
            int pool_used = 0;
            if (compact_ok && gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                pool_used = ds4_gpu_try_load_layer_experts_to_pool(model_map,
                                                                   layer_index,
                                                                   selectedbuf,
                                                                   selected_off,
                                                                   n_expert,
                                                                   n_active,
                                                                   active_ids,
                                                                   gate_offset,
                                                                   up_offset,
                                                                   down_offset,
                                                                   gate_expert_bytes,
                                                                   down_expert_bytes,
                                                                   n_total_expert,
                                                                   &gate_buf,
                                                                   &up_buf,
                                                                   &down_buf,
                                                                   &source_n_total_expert);
            }
            if (compact_ok && !pool_used && gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                ds4_gpu_expert_source_cache_note(model_map,
                                                 layer_index,
                                                 n_active,
                                                 active_ids,
                                                 gate_offset,
                                                 up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes);
            }
            if (io_profile) ds4_gpu_expert_io_prof_reset();
            const double gather_t0 = io_profile ? ds4_gpu_now_ms() : 0.0;
            const double copy_t0 = (!pool_used && ds4_gpu_expert_profile_is_enabled()) ? ds4_gpu_now_ms() : 0.0;
            int load_ok = compact_ok && (pool_used ||
                          ds4_gpu_load_layer_experts_to_scratch(model_map,
                                                                 layer_index,
                                                                 n_active,
                                                                 active_ids,
                                                                 gate_offset,
                                                                 up_offset,
                                                                 down_offset,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 n_total_expert));
            if (io_profile) {
                ds4_gpu_expert_io_prof_report("decode",
                                              pool_used ? "pool" : "gather",
                                              layer_index,
                                              n_active,
                                              1u,
                                              ds4_gpu_now_ms() - gather_t0,
                                              drain_ms);
            }
            if (!pool_used && compact_ok && ds4_gpu_expert_profile_is_enabled()) {
                ds4_gpu_expert_profile_record(layer_index,
                                              active_ids,
                                              n_active,
                                              n_total_expert,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              n_expert,
                                              ds4_gpu_now_ms() - copy_t0);
            }
            if (!load_ok) {
                if (was_batched) (void)ds4_gpu_begin_commands();
                return 0;
            }
            if (!pool_used) {
                gate_buf = g_moe_scratch_gate;
                up_buf = g_moe_scratch_up;
                down_buf = g_moe_scratch_down;
                source_n_total_expert = n_active;
            }
            gate_inner = 0;
            up_inner = 0;
            down_inner = 0;
            if (was_batched && ds4_gpu_begin_commands() == 0) return 0;
        }

        const uint32_t n_tokens = 1;
        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, source_n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, source_n_total_expert,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        int ok = 1;
        const bool write_clamped_moe =
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL;
        id<MTLComputePipelineState> pair_swiglu_pipeline = nil;
        if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
        } else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
        }
        const bool fuse_pair_swiglu =
            !g_quality_mode &&
            !write_clamped_moe &&
            getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
            pair_swiglu_pipeline != nil;
        if (fuse_pair_swiglu) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_id_pair_swiglu(cb,
                                                        pair_swiglu_pipeline,
                                                        &gate_args,
                                                        &act_args,
                                                        gate_buf,
                                                        (NSUInteger)gate_inner,
                                                        up_buf,
                                                        (NSUInteger)up_inner,
                                                        xbuf,
                                                        ds4_gpu_tensor_offset(x),
                                                        gatebuf,
                                                        ds4_gpu_tensor_offset(gate),
                                                        upbuf,
                                                        ds4_gpu_tensor_offset(up),
                                                        midbuf,
                                                        ds4_gpu_tensor_offset(mid),
                                                        selectedbuf,
                                                        ds4_gpu_tensor_offset(selected),
                                                        weightsbuf,
                                                        ds4_gpu_tensor_offset(weights),
                                                        gate_smem,
                                                        2,
                                                        false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_iq2_xxs_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_Q4_K &&
                   g_moe_mul_mv_id_q4_k_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_q4_k_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            gate_buf,
                                            (NSUInteger)gate_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            gatebuf,
                                            ds4_gpu_tensor_offset(gate),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            up_buf,
                                            (NSUInteger)up_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            upbuf,
                                            ds4_gpu_tensor_offset(up),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false);
        }
        if (ok && !fuse_pair_swiglu) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    false);
        }

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            n_expert == 6 &&
            n_tokens == 1 &&
            down_sum6_pipeline != nil;
        if (ok && direct_down_sum) {
            ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                 down_sum6_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2);
        } else if (ok) {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                 down_mv_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 down_dst,
                                                 down_dst_off,
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2,
                                                 false);
        }
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed tensor MoE")) return 0;
    }

    return 1;
}

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert * sizeof(int);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        g_wrap_mlock_suppress = 1;   /* routed expert tensors: gathered, never wired */
        id<MTLBuffer> gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
        id<MTLBuffer> up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
        id<MTLBuffer> down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
        g_wrap_mlock_suppress = 0;
        if (!gate_buf || !up_buf || !down_buf) return 0;
        uint32_t source_n_total_expert = n_total_expert;

        /* DS4_METAL_EXPERT_OFFLOAD is meant for the full q2 target model, whose
         * routed experts are deliberately non-resident and must be gathered into
         * compact scratch/pool before the MoE kernels index them.  The MTP support
         * model is mapped as a fully resident model range; forcing its Q4_K routed
         * experts through the A3 scratch path just copies resident bytes again and
         * made distributed MTP a net slowdown.  Leave non-q2 routed tensors on the
         * direct resident-buffer path. */
        const bool a3_expert_offload =
            ds4_gpu_expert_offload_enabled() &&
            !ds4_gpu_expert_offload_direct_enabled() &&
            gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
            down_type == DS4_METAL_TENSOR_Q2_K;
        if (a3_expert_offload) {
            const int io_profile = ds4_gpu_expert_io_profile_enabled();
            /* Prefill-sized batch bytes are one-shot first-touch reads: keep
             * them out of the page cache.  Verify rounds (kc<=16) stay cached
             * -- their unions overlap decode-hot experts (wave 25 A/B). */
            g_expert_gather_nocache_call =
                (n_tokens >= ds4_gpu_expert_batch_nocache_min_tokens()) &&
                ds4_gpu_expert_batch_nocache_enabled();
            const int was_batched = (g_batch_cb != nil);
            double drain_ms = 0.0;
            if (was_batched) {
                const double drain_t0 = io_profile ? ds4_gpu_now_ms() : 0.0;
                if (ds4_gpu_expert_drain_commands("routed MoE drain") == 0) return 0;
                if (io_profile) drain_ms = ds4_gpu_now_ms() - drain_t0;
            }
            /* Wave 36: batch union prediction.  The batch path never enqueued
             * prefetch jobs at all (only decode did, and with one row): verify
             * gathers ran unpredicted/cold.  Snapshot the drained batch hidden
             * and predict the NEXT layer's expert union; advisories land in
             * the next drain's idle-disk window.  Cached-fd batches only --
             * NOCACHE preads bypass the page cache, warming is useless there. */
            if (ds4_gpu_expert_prefetch_enabled() &&
                !g_expert_gather_nocache_call &&
                n_tokens > 1u &&
                xbuf.storageMode == MTLStorageModeShared) {
                const float *x_cpu = (const float *)((const uint8_t *)xbuf.contents +
                                                     (size_t)ds4_gpu_tensor_offset(x));
                ds4_gpu_expert_prefetch_enqueue_batch(layer_index + 1u, x_cpu,
                                                      expert_in_dim, n_tokens);
            }
            uint32_t active_ids[1024];
            uint32_t n_active = 0;
            const uint64_t total_picks_u64 = (uint64_t)n_tokens * n_expert;
            if (total_picks_u64 > UINT32_MAX) {
                if (was_batched) (void)ds4_gpu_begin_commands();
                return 0;
            }
            /* Pass 1 only: count the active set first so we can choose between
             * per-expert gather and P1.2 full-layer streaming before the slot
             * remap rewrites the selected-id buffer. */
            int compact_ok = ds4_gpu_collect_active_experts(selectedbuf,
                                                            selected_off,
                                                            (uint32_t)total_picks_u64,
                                                            n_total_expert,
                                                            active_ids,
                                                            1024,
                                                            &n_active);
            if (io_profile) ds4_gpu_expert_io_prof_reset();
            const double gather_t0 = io_profile ? ds4_gpu_now_ms() : 0.0;
            /* P1.2: a big prefill chunk activates nearly every expert of the
             * layer, so ~256 scattered reads degenerate to random IO.  Stream
             * the whole gate/up/down tensors sequentially instead and keep the
             * original expert ids (selected buffer untouched, kernels index
             * the full-layer scratch exactly like the resident direct path). */
            int stream_used = 0;
            if (compact_ok && n_tokens > 1u && !g_expert_pool_warm_batch &&
                gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K &&
                ds4_gpu_expert_stream_enabled() &&
                !ds4_gpu_expert_stream_rfetch_bypass() &&
                (uint64_t)n_active * 100ull >=
                    (uint64_t)n_total_expert * ds4_gpu_expert_stream_threshold_pct()) {
                stream_used = ds4_gpu_stream_layer_experts_to_scratch(model_map,
                                                                      gate_offset,
                                                                      up_offset,
                                                                      down_offset,
                                                                      gate_expert_bytes,
                                                                      down_expert_bytes,
                                                                      n_total_expert);
            }
            if (compact_ok && !stream_used) {
                compact_ok = ds4_gpu_remap_selected_to_slots(selectedbuf,
                                                             selected_off,
                                                             (uint32_t)total_picks_u64,
                                                             n_total_expert,
                                                             active_ids,
                                                             n_active);
            }
            int pool_used = 0;
            if (!stream_used && compact_ok && g_expert_pool_warm_batch &&
                gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                pool_used = ds4_gpu_try_load_layer_experts_to_pool(model_map,
                                                                   layer_index,
                                                                   selectedbuf,
                                                                   selected_off,
                                                                   (uint32_t)total_picks_u64,
                                                                   n_active,
                                                                   active_ids,
                                                                   gate_offset,
                                                                   up_offset,
                                                                   down_offset,
                                                                   gate_expert_bytes,
                                                                   down_expert_bytes,
                                                                   n_total_expert,
                                                                   &gate_buf,
                                                                   &up_buf,
                                                                   &down_buf,
                                                                   &source_n_total_expert);
            }
            if (!stream_used && compact_ok && !pool_used &&
                gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                down_type == DS4_METAL_TENSOR_Q2_K) {
                ds4_gpu_expert_source_cache_note(model_map,
                                                 layer_index,
                                                 n_active,
                                                 active_ids,
                                                 gate_offset,
                                                 up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes);
            }
            const double copy_t0 = (!pool_used && !stream_used && ds4_gpu_expert_profile_is_enabled()) ?
                ds4_gpu_now_ms() : 0.0;
            int load_ok = compact_ok && (stream_used || pool_used ||
                          ds4_gpu_load_layer_experts_to_scratch(model_map,
                                                                 layer_index,
                                                                 n_active,
                                                                 active_ids,
                                                                 gate_offset,
                                                                 up_offset,
                                                                 down_offset,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 n_total_expert));
            if (io_profile) {
                ds4_gpu_expert_io_prof_report("batch",
                                              stream_used ? "stream" :
                                              (pool_used ? "pool" : "gather"),
                                              layer_index,
                                              n_active,
                                              n_tokens,
                                              ds4_gpu_now_ms() - gather_t0,
                                              drain_ms);
            }
            if (!pool_used && !stream_used && compact_ok && ds4_gpu_expert_profile_is_enabled()) {
                ds4_gpu_expert_profile_record(layer_index,
                                              active_ids,
                                              n_active,
                                              n_total_expert,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              (uint32_t)total_picks_u64,
                                              ds4_gpu_now_ms() - copy_t0);
            }
            if (!load_ok) {
                if (was_batched) (void)ds4_gpu_begin_commands();
                return 0;
            }
            if (!pool_used) {
                gate_buf = g_moe_scratch_gate;
                up_buf = g_moe_scratch_up;
                down_buf = g_moe_scratch_down;
                source_n_total_expert = stream_used ? n_total_expert : n_active;
            }
            gate_inner = 0;
            up_inner = 0;
            down_inner = 0;
            if (was_batched && ds4_gpu_begin_commands() == 0) return 0;
        }

        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        id<MTLComputePipelineState> gate_mm_pipeline = nil;
        id<MTLComputePipelineState> up_mm_pipeline = nil;
        id<MTLComputePipelineState> down_mm_pipeline = nil;
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed batch MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, source_n_total_expert,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, source_n_total_expert,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);
        /*
         * Grouped-GEMM threshold. The mm_id path maps token-rows per expert
         * and reads each active expert's weights exactly ONCE (expert-major),
         * while the mv/pair paths re-read the expert weights once per
         * (expert,token) pair: a 12-token verify batch re-reads ~486MiB of
         * scratch instead of the ~270MiB union, through matvec-shaped kernels
         * with much lower effective bandwidth. The old >=32 threshold was
         * tuned when the only small batches were MTP's 2-token suffixes; PC.1
         * copy-speculation batches are 5..12 tokens and sit right in the slow
         * gap. Default lowered to 8 (A/B: DS4_METAL_MOE_MM_ID_MIN=32 restores
         * the old split, =0 forces never).
         */
        static uint32_t mm_id_min_cached;
        static int mm_id_min_init;
        if (!mm_id_min_init) {
            uint64_t v = ds4_gpu_env_u64("DS4_METAL_MOE_MM_ID_MIN", 8u);
            if (v == 0) v = UINT32_MAX;     /* 0 = never use mm_id */
            if (v < 2u) v = 2u;
            mm_id_min_cached = (uint32_t)(v > UINT32_MAX ? UINT32_MAX : v);
            mm_id_min_init = 1;
        }
        const bool use_mm_id = n_tokens >= mm_id_min_cached &&
                               ds4_gpu_mul_mm_id_map0_name(n_expert) != NULL;
        /*
         * Speculative verification is neither normal decode nor large prefill:
         * the target model must verify a small suffix in one layer-major pass.
         * For that shape the prefill expert-major GEMM path is too large, but
         * the decode pair kernels are exactly the right primitive: they read
         * the same activation once and compute routed gate/up together for
         * every selected expert row, and their dispatch grid is
         * (n_expert x n_tokens) pairs - shape generic.
         *
         * PC.1 copy speculation widened the verify batches from MTP's 2 to
         * 5..12 tokens, which used to fall through to the generic split-mv
         * path: measured (2026-06-10 code-edit run) drain 100-256 ms/layer for
         * a 12-token batch = ~5-10x worse per token than the >=32-token
         * grouped GEMM (1.3 ms/token/layer). Route everything below the GEMM
         * threshold through the pair kernels instead; cap adjustable for A/B
         * (DS4_METAL_MOE_TINY_PAIR_MAX=4 restores the old behavior).
         */
        static uint32_t tiny_pair_max_cached;
        static int tiny_pair_max_init;
        if (!tiny_pair_max_init) {
            uint64_t v = ds4_gpu_env_u64("DS4_METAL_MOE_TINY_PAIR_MAX", 16u);
            if (v > 31u) v = 31u;
            tiny_pair_max_cached = (uint32_t)v;
            tiny_pair_max_init = 1;
        }
        const bool use_tiny_pair_mv =
            !g_quality_mode &&
            n_tokens <= tiny_pair_max_cached &&
            !use_mm_id &&
            ((gate_type == DS4_METAL_TENSOR_IQ2_XXS && g_moe_mul_mv_id_iq2_xxs_pair_pipeline) ||
             (gate_type == DS4_METAL_TENSOR_Q4_K && g_moe_mul_mv_id_q4_k_pair_pipeline));
        ds4_gpu_mul_mm_id_map_args gate_map_args = { 0 };
        ds4_gpu_mul_mm_id_args gate_mm_args = { 0 };
        ds4_gpu_mul_mm_id_args down_mm_args = { 0 };
        id<MTLComputePipelineState> map_pipeline = nil;
        /*
         * The grouped routed-MoE matmul loads activation tiles as half before
         * using SIMD-group MMA.  Store the SwiGLU/route-weight intermediate in
         * that same precision so the down projection avoids a large F32 mid
         * write/read. --quality keeps the older F32 intermediate.
         */
        const bool request_mid_f16 = !g_quality_mode;
        if (use_mm_id) {
            gate_map_args =
                ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, source_n_total_expert, 1, n_expert, n_tokens);
            gate_mm_args =
                ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, source_n_total_expert,
                                              gate_row_bytes, gate_expert_bytes,
                                              1, n_expert, n_tokens);
            down_mm_args =
                ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, source_n_total_expert,
                                                        down_row_bytes, down_expert_bytes,
                                                        n_expert, n_expert, n_tokens,
                                                        request_mid_f16 ? sizeof(uint16_t) : sizeof(float));

            map_pipeline = ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
            gate_mm_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
            up_mm_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
            down_mm_pipeline = request_mid_f16 ?
                ds4_gpu_routed_mm_f16_rhs_pipeline(down_type) :
                ds4_gpu_routed_mm_pipeline(down_type);
            if (!map_pipeline || !gate_mm_pipeline || !up_mm_pipeline || !down_mm_pipeline) {
                return 0;
            }
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        const bool moe_stage_profile =
            getenv("DS4_METAL_MOE_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        const char *moe_stage_filter = getenv("DS4_METAL_MOE_STAGE_PROFILE_FILTER");
        const char *moe_path =
            use_mm_id ? "mm_id" :
            (use_tiny_pair_mv ? "tiny_pair_mv" : "mv");
        double moe_stage_t0 = moe_stage_profile ? ds4_gpu_now_ms() : 0.0;
        if (moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            moe_stage_t0 = ds4_gpu_now_ms();
        }
#define DS4_METAL_PROFILE_MOE_STAGE(name) do { \
            if (ok && moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const char *stage_name = (name); \
                    const double now_ms = ds4_gpu_now_ms(); \
                    const int print_stage = \
                        !moe_stage_filter || !moe_stage_filter[0] || \
                        strstr(stage_name, moe_stage_filter) != NULL; \
                    if (print_stage) { \
                        fprintf(stderr, \
                                "ds4: Metal routed MoE stage layer=%u tokens=%u pairs=%u experts=%u " \
                                "gate=%s down=%s path=%s mid=%s %s=%.3f ms\n", \
                                layer_index, n_tokens, pair_rows, n_expert, \
                                ds4_gpu_metal_tensor_type_name(gate_type), \
                                ds4_gpu_metal_tensor_type_name(down_type), \
                                moe_path, \
                                request_mid_f16 ? "f16" : "f32", \
                                stage_name, now_ms - moe_stage_t0); \
                    } \
                    moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            !use_mm_id &&
            n_expert == 6 &&
            n_tokens <= 4u &&
            down_sum6_pipeline != nil;
        int ok = 0;
        if (use_mm_id) {
            /*
             * The routed pair ids are the same for gate, up, and down. Build
             * the expert-major work map once, then reuse it for all three
             * batched expert matmuls.
             */
            ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                                map_pipeline,
                                                &gate_map_args,
                                                &gate_mm_args,
                                                selectedbuf,
                                                ds4_gpu_tensor_offset(selected));
            DS4_METAL_PROFILE_MOE_STAGE("map");
            if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                           gate_mm_pipeline,
                                                           &gate_mm_args,
                                                           gate_buf,
                                                           (NSUInteger)gate_inner,
                                                           xbuf,
                                                           ds4_gpu_tensor_offset(x),
                                                           gatebuf,
                                                           ds4_gpu_tensor_offset(gate));
                DS4_METAL_PROFILE_MOE_STAGE("gate");
            }
            if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                   up_mm_pipeline,
                                                   &gate_mm_args,
                                                   up_buf,
                                                   (NSUInteger)up_inner,
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up));
                DS4_METAL_PROFILE_MOE_STAGE("up");
            }
        } else if (use_tiny_pair_mv) {
            id<MTLComputePipelineState> pair_pipeline =
                gate_type == DS4_METAL_TENSOR_IQ2_XXS ?
                    g_moe_mul_mv_id_iq2_xxs_pair_pipeline :
                    g_moe_mul_mv_id_q4_k_pair_pipeline;
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  gate_buf,
                                                  (NSUInteger)gate_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  gatebuf,
                                                  ds4_gpu_tensor_offset(gate),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  up_buf,
                                                  (NSUInteger)up_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  upbuf,
                                                  ds4_gpu_tensor_offset(up),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false);
        }
        DS4_METAL_PROFILE_MOE_STAGE("gate_up");
        const bool use_fused_activation = !g_quality_mode;
        const bool use_mid_f16 =
            use_mm_id &&
            use_fused_activation &&
            request_mid_f16;
        if (mid_is_f16) *mid_is_f16 = use_mid_f16;
        if (ok && use_fused_activation) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    use_mid_f16);
        } else if (ok && clamp > 1.0e-6f) {
            ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -FLT_MAX,
                                                 clamp);
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                     g_unary_silu_pipeline,
                                                     gatebuf,
                                                     ds4_gpu_tensor_offset(gate),
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     expert_mid_dim,
                                                     pair_rows,
                                                     1,
                                                     0.0f,
                                                     0.0f);
            }
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -clamp,
                                                 clamp);
            }
            if (ok) {
                ds4_gpu_bin_args mul_args =
                    ds4_gpu_make_bin_same_rows_args(expert_mid_dim, pair_rows);
                ok = ds4_gpu_encode_bin_f32_rows(cb,
                                                   g_mul_pipeline,
                                                   &mul_args,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up),
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid));
            }
        } else if (ok) {
            ok = ds4_gpu_encode_swiglu_flat(cb,
                                              gatebuf,
                                              ds4_gpu_tensor_offset(gate),
                                              upbuf,
                                              ds4_gpu_tensor_offset(up),
                                              midbuf,
                                              ds4_gpu_tensor_offset(mid),
                                              (uint32_t)((uint64_t)pair_rows * expert_mid_dim));
        }
        if (ok && !use_fused_activation) {
            ds4_gpu_bin_args weight_args =
                ds4_gpu_make_bin_rowwise_scalar_args(expert_mid_dim, pair_rows);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_bin_mul_scalar_pipeline,
                                               &weight_args,
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid),
                                               weightsbuf,
                                               ds4_gpu_tensor_offset(weights),
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid));
        }
        DS4_METAL_PROFILE_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        if (ok) {
            if (direct_down_sum) {
                ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                     down_sum6_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     outbuf,
                                                     ds4_gpu_tensor_offset(out),
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     2);
            } else if (use_mm_id) {
                ok = ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                       down_mm_pipeline,
                                                       &down_mm_args,
                                                       down_buf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off);
            } else {
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                     down_mv_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     down_dst,
                                                     down_dst_off,
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     2,
                                                     false);
            }
        }
        DS4_METAL_PROFILE_MOE_STAGE("down");
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        DS4_METAL_PROFILE_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed batch MoE")) return 0;
#undef DS4_METAL_PROFILE_MOE_STAGE
    }

    return 1;
}

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_hc == 0 || n_hc > 16) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t mix_tensor_bytes = ds4_gpu_tensor_bytes(mix);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (!mixbuf || !outbuf ||
            mix_tensor_bytes < mix_bytes ||
            out_tensor_bytes < mix_bytes) {
            fprintf(stderr, "ds4: Metal HC split received undersized activation buffers\n");
            return 0;
        }
        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset) {
            fprintf(stderr, "ds4: Metal HC split parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        if (!scalebuf || !basebuf) return 0;

        uint64_t n_rows64 = mix_tensor_bytes / mix_bytes;
        const uint64_t out_rows64 = out_tensor_bytes / mix_bytes;
        if (out_rows64 < n_rows64) n_rows64 = out_rows64;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC split row count is outside supported range\n");
            return 0;
        }

        ds4_gpu_hc_split_args args = {
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb01 = mix_bytes,
            .nb1 = mix_bytes,
            .eps = eps,
        };
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_rows64));
        const NSUInteger n_tg = ((NSUInteger)n_rows64 + nth - 1u) / nth;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_split_sinkhorn_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sinkhorn")) return 0;
    }

    return 1;
}

static int ds4_gpu_hc_weighted_sum_strided(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint64_t                weight_offset,
        uint64_t                weight_row_stride,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0 ||
        weight_row_stride < (uint64_t)n_hc * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes || out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC weighted sum output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / out_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC weighted sum token count is outside supported range\n");
            return 0;
        }

        const uint64_t x_row_values = (uint64_t)n_hc * n_embd;
        if (x_row_values == 0 ||
            x_row_values > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (x_row_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / ((uint64_t)n_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC weighted sum activation size overflow\n");
            return 0;
        }

        const uint64_t x_bytes = n_tokens64 * x_row_values * sizeof(float);
        const uint64_t w_last = weight_offset +
                                (n_tokens64 - 1u) * weight_row_stride +
                                (uint64_t)n_hc * sizeof(float);
        if (!xbuf || !wbuf || !outbuf ||
            ds4_gpu_tensor_bytes(residual_hc) < x_bytes ||
            ds4_gpu_tensor_bytes(weights) < w_last) {
            fprintf(stderr, "ds4: Metal HC weighted sum received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_weighted_sum_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_w0 = sizeof(float),
            .nb_w1 = weight_row_stride,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
        };
        const uint64_t n_elem = (uint64_t)n_embd * n_tokens64;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_weighted_sum_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) + (NSUInteger)weight_offset atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, label)) return 0;
    }

    return 1;
}

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return ds4_gpu_hc_weighted_sum_strided(out,
                                             residual_hc,
                                             weights,
                                             0,
                                             (uint64_t)n_hc * sizeof(float),
                                             n_embd,
                                             n_hc,
                                             "HC weighted sum");
}

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return ds4_gpu_hc_weighted_sum_strided(out,
                                             residual_hc,
                                             split,
                                             0,
                                             mix_hc * sizeof(float),
                                             n_embd,
                                             n_hc,
                                             "HC weighted sum split");
}

/* Release decode fused HC pre-sublayer operation.  The graph driver owns the
 * optional reference fallback so this function stays a direct fused dispatch. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc == 0) {
        return 0;
    }
    if (n_hc != 4) {
        fprintf(stderr, "ds4: Metal fused HC split/sum is specialized for HC=4\n");
        return 0;
    }

    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
            out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal fused HC split/sum output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_rows64 = out_tensor_bytes / out_row_bytes;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX ||
            n_rows64 > UINT64_MAX / mix_bytes ||
            n_rows64 > UINT64_MAX / residual_row_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum row count is outside supported range\n");
            return 0;
        }

        const uint64_t mix_total_bytes = n_rows64 * mix_bytes;
        const uint64_t residual_total_bytes = n_rows64 * residual_row_bytes;
        if (!mixbuf || !splitbuf || !xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(mix) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(split) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < residual_total_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum received undersized activation buffers\n");
            return 0;
        }

        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset) {
            fprintf(stderr, "ds4: Metal fused HC split/sum parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        if (!scalebuf || !basebuf) return 0;

        ds4_gpu_hc_split_weighted_sum_args args = {
            .n_embd = (int64_t)n_embd,
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb_mix1 = mix_bytes,
            .nb_split1 = mix_bytes,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = residual_row_bytes,
            .nb0 = sizeof(float),
            .nb1 = out_row_bytes,
            .eps = eps,
        };

        NSUInteger nth = g_hc_split_weighted_sum_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n_embd) nth = (NSUInteger)n_embd;
        if (nth == 0) nth = 1u;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_split_weighted_sum_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:4];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_hc * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows64, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sum fused")) return 0;
    }

    return 1;
}

/* Decode-only HC-pre plus the immediately following weighted RMSNorm.  This is
 * intentionally specialized for DS4's fixed HC=4, embd=4096 shape; larger
 * batched prefill keeps using the existing two-stage path. */
int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd != 4096 || n_hc != 4) {
        return 0;
    }

    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> normbuf = ds4_gpu_tensor_buffer(norm_out);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
            out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_rows64 = out_tensor_bytes / out_row_bytes;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX ||
            n_rows64 > UINT64_MAX / mix_bytes ||
            n_rows64 > UINT64_MAX / residual_row_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm row count is outside supported range\n");
            return 0;
        }

        const uint64_t mix_total_bytes = n_rows64 * mix_bytes;
        const uint64_t residual_total_bytes = n_rows64 * residual_row_bytes;
        const uint64_t out_total_bytes = n_rows64 * out_row_bytes;
        if (!mixbuf || !splitbuf || !xbuf || !outbuf || !normbuf ||
            ds4_gpu_tensor_bytes(mix) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(split) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < residual_total_bytes ||
            ds4_gpu_tensor_bytes(norm_out) < out_total_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm received undersized activation buffers\n");
            return 0;
        }

        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size || out_row_bytes > model_size - norm_weight_offset) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        uint64_t norm_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        id<MTLBuffer> normwbuf = ds4_gpu_wrap_model_range(model_map, model_size, norm_weight_offset, out_row_bytes, &norm_inner);
        if (!scalebuf || !basebuf || !normwbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_hc_split_weighted_sum_norm_pipeline,
                                   "kernel_dsv4_hc_split_weighted_sum_norm4");
        if (!pipeline) return 0;

        ds4_gpu_hc_split_weighted_sum_norm_args args = {
            .n_embd = (int64_t)n_embd,
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb_mix1 = mix_bytes,
            .nb_split1 = mix_bytes,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = residual_row_bytes,
            .nb0 = sizeof(float),
            .nb1 = out_row_bytes,
            .nb_norm1 = out_row_bytes,
            .eps = eps,
            .norm_eps = norm_eps,
        };

        NSUInteger nth = ds4_gpu_rms_norm_threads(n_embd);
        if (nth > pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm requires %lu threads but pipeline supports %lu\n",
                    (unsigned long)nth,
                    (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:4];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc setBuffer:normwbuf offset:(NSUInteger)norm_inner atIndex:7];
        [enc setBuffer:normbuf offset:ds4_gpu_tensor_offset(norm_out) atIndex:8];
        [enc setThreadgroupMemoryLength:((NSUInteger)n_embd + 4u + 32u) * sizeof(float)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows64, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sum/norm fused")) return 0;
    }

    return 1;
}

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !pre || !model_map || n_hc == 0) return 0;

    @autoreleasepool {
        if ((n_hc % 4u) != 0) {
            fprintf(stderr, "ds4: Metal output HC weights requires a multiple-of-4 HC width\n");
            return 0;
        }

        id<MTLBuffer> prebuf = ds4_gpu_tensor_buffer(pre);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (row_bytes == 0 || out_tensor_bytes < row_bytes || out_tensor_bytes % row_bytes != 0) {
            fprintf(stderr, "ds4: Metal output HC weights size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX ||
            n_tokens64 > UINT64_MAX / row_bytes) {
            fprintf(stderr, "ds4: Metal output HC weights token count is outside supported range\n");
            return 0;
        }

        const uint64_t bytes = n_tokens64 * row_bytes;
        if (!prebuf || !outbuf ||
            ds4_gpu_tensor_bytes(pre) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal output HC weights received undersized buffers\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                            scale_offset, sizeof(float),
                                                            &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           base_offset, row_bytes,
                                                           &base_inner);
        if (!scalebuf || !basebuf) return 0;

        const uint32_t n_tokens = (uint32_t)n_tokens64;
        ds4_gpu_bin_args mul_args = ds4_gpu_make_bin_rows_args(n_hc, n_tokens, 1);
        ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(n_hc, n_tokens, n_hc);
        ds4_gpu_unary_args sigmoid_args = ds4_gpu_make_unary_rows_args(n_hc, n_tokens, 1, 0.0f, 0.0f);
        ds4_gpu_unary_args scale_args = ds4_gpu_make_unary_rows_args(n_hc, n_tokens, 1, 1.0f, eps);

        NSUInteger mul_nth_max = g_bin_mul_scalar_pipeline.maxTotalThreadsPerThreadgroup;
        if (mul_nth_max > 256u) mul_nth_max = 256u;
        NSUInteger mul_nth = 1u;
        while (2u * mul_nth < (NSUInteger)mul_args.ne0 && mul_nth < mul_nth_max) {
            mul_nth *= 2u;
        }

        NSUInteger add_nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (add_nth_max > 256u) add_nth_max = 256u;
        NSUInteger add_nth = 1u;
        while (2u * add_nth < (NSUInteger)add_args.ne0 && add_nth < add_nth_max) {
            add_nth *= 2u;
        }

        NSUInteger unary_nth_max = g_unary_sigmoid_pipeline.maxTotalThreadsPerThreadgroup;
        if (unary_nth_max > 256u) unary_nth_max = 256u;
        NSUInteger unary_nth = (NSUInteger)sigmoid_args.ne00;
        if (unary_nth > unary_nth_max) unary_nth = unary_nth_max;
        if (unary_nth == 0) unary_nth = 1u;
        const NSUInteger unary_nk0 = ((NSUInteger)sigmoid_args.ne00 + unary_nth - 1u) / unary_nth;
        const NSUInteger out_offset = ds4_gpu_tensor_offset(out);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);

        [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
        [enc setBytes:&mul_args length:sizeof(mul_args) atIndex:0];
        [enc setBuffer:prebuf offset:ds4_gpu_tensor_offset(pre) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:outbuf offset:out_offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mul_args.ne01,
                                              (NSUInteger)mul_args.ne02,
                                              (NSUInteger)mul_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(mul_nth, 1, 1)];

        [enc setComputePipelineState:g_add_pipeline];
        [enc setBytes:&add_args length:sizeof(add_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:2];
        [enc setBuffer:outbuf offset:out_offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)add_args.ne01,
                                              (NSUInteger)add_args.ne02,
                                              (NSUInteger)add_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(add_nth, 1, 1)];

        [enc setComputePipelineState:g_unary_sigmoid_pipeline];
        [enc setBytes:&sigmoid_args length:sizeof(sigmoid_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:outbuf offset:out_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(unary_nk0 * (NSUInteger)sigmoid_args.ne01,
                                              (NSUInteger)sigmoid_args.ne02,
                                              (NSUInteger)sigmoid_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(unary_nth, 1, 1)];

        [enc setComputePipelineState:g_unary_scale_pipeline];
        [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:outbuf offset:out_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(unary_nk0 * (NSUInteger)scale_args.ne01,
                                              (NSUInteger)scale_args.ne02,
                                              (NSUInteger)scale_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(unary_nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "output HC weights")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> postbuf = ds4_gpu_tensor_buffer(post);
        id<MTLBuffer> combbuf = ds4_gpu_tensor_buffer(comb);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t comb_values = (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            comb_values > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (comb_values * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t post_bytes = n_tokens64 * (uint64_t)n_hc * sizeof(float);
        const uint64_t comb_bytes = n_tokens64 * comb_values * sizeof(float);
        if (!blockbuf || !resbuf || !postbuf || !combbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(post) < post_bytes ||
            ds4_gpu_tensor_bytes(comb) < comb_bytes) {
            fprintf(stderr, "ds4: Metal HC expand received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = (uint64_t)n_hc * n_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:postbuf offset:ds4_gpu_tensor_offset(post) atIndex:3];
        [enc setBuffer:combbuf offset:ds4_gpu_tensor_offset(comb) atIndex:4];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand split output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand split token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            mix_hc > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (mix_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand split activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t split_bytes = n_tokens64 * mix_hc * sizeof(float);
        if (!blockbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes) {
            fprintf(stderr, "ds4: Metal HC expand split received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:3];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:4];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand split")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> addbuf = ds4_gpu_tensor_buffer(block_add);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand add split output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand add split token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            mix_hc > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (mix_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand add split activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t split_bytes = n_tokens64 * mix_hc * sizeof(float);
        if (!blockbuf || !addbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(block_add) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes) {
            fprintf(stderr, "ds4: Metal HC expand add split received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 1,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:3];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:4];
        [enc setBuffer:addbuf offset:ds4_gpu_tensor_offset(block_add) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand add split")) return 0;
    }

    return 1;
}

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out ||
        !residual_hc || !split || n_embd == 0 || n_hc == 0 ||
        n_hc != 4 || out_dim != n_embd || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(shared_mid);
        id<MTLBuffer> sharedbuf = ds4_gpu_tensor_buffer(shared_out);
        id<MTLBuffer> routedbuf = ds4_gpu_tensor_buffer(routed_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        const uint64_t shared_mid_bytes = in_dim * sizeof(float);
        const uint64_t embd_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t split_bytes = mix_hc * sizeof(float);

        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal shared-down HC fusion weight range is outside the mapped model\n");
            return 0;
        }
        if (!midbuf || !sharedbuf || !routedbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(shared_mid) < shared_mid_bytes ||
            ds4_gpu_tensor_bytes(shared_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(routed_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes ||
            ds4_gpu_tensor_bytes(out_hc) < hc_bytes) {
            fprintf(stderr, "ds4: Metal shared-down HC fusion received undersized buffers\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes,
                                                        &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        mv_args.nr0 = mv_dispatch.nr0;

        ds4_gpu_hc_expand_args hc_args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = 1,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 1,
        };

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_shared_down_hc_expand4_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&hc_args length:sizeof(hc_args) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(shared_mid) atIndex:3];
        [enc setBuffer:sharedbuf offset:ds4_gpu_tensor_offset(shared_out) atIndex:4];
        [enc setBuffer:routedbuf offset:ds4_gpu_tensor_offset(routed_out) atIndex:5];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:6];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:7];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:8];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:9];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                              (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "shared-down HC expand fused")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !model_map || !x || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0 || n_hc != 4 || out_dim != n_embd ||
        (in_dim & 31u) != 0 || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t embd_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t split_bytes = mix_hc * sizeof(float);

        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8 HC fusion weight range is outside the mapped model\n");
            return 0;
        }
        if (!xbuf || !blockbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(block_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes ||
            ds4_gpu_tensor_bytes(out_hc) < hc_bytes) {
            fprintf(stderr, "ds4: Metal Q8 HC fusion received undersized buffers\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes,
                                                        &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        mv_args.nr0 = mv_dispatch.nr0;

        ds4_gpu_hc_expand_args hc_args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = 1,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_q8_hc_expand4_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&hc_args length:sizeof(hc_args) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:4];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:5];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:6];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:7];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:8];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                              (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8 HC expand fused")) return 0;
    }

    return 1;
}
