/* metal_state.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

id<MTLDevice> g_device;

id<MTLCommandQueue> g_queue;

id<MTLLibrary> g_library;

id<MTLCommandBuffer> g_batch_cb;

id<MTLComputeCommandEncoder> g_batch_enc;

NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;

id<MTLComputePipelineState> g_set_rows_f32_i32_pipeline;

id<MTLComputePipelineState> g_get_rows_f32_pipeline;

id<MTLComputePipelineState> g_get_rows_f16_pipeline;

id<MTLComputePipelineState> g_get_rows_i32_pipeline;

id<MTLComputePipelineState> g_repeat_f32_pipeline;

id<MTLComputePipelineState> g_concat_pipeline;

id<MTLComputePipelineState> g_cpy_f32_f32_pipeline;

id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;

id<MTLComputePipelineState> g_cpy_f16_f32_pipeline;

id<MTLComputePipelineState> g_swiglu_pipeline;

id<MTLComputePipelineState> g_add_pipeline;

id<MTLComputePipelineState> g_moe_sum6_pipeline;

id<MTLComputePipelineState> g_mul_pipeline;

id<MTLComputePipelineState> g_rms_norm_pipeline;

id<MTLComputePipelineState> g_rms_norm_plain_pipeline;

id<MTLComputePipelineState> g_dsv4_qkv_rms_norm_pipeline;

id<MTLComputePipelineState> g_hc_split_sinkhorn_pipeline;

id<MTLComputePipelineState> g_hc_split_weighted_sum_pipeline;

id<MTLComputePipelineState> g_hc_split_weighted_sum_norm_pipeline;

id<MTLComputePipelineState> g_hc_weighted_sum_pipeline;

id<MTLComputePipelineState> g_hc_expand_pipeline;

id<MTLComputePipelineState> g_unary_sigmoid_pipeline;

id<MTLComputePipelineState> g_unary_silu_pipeline;

id<MTLComputePipelineState> g_unary_softplus_pipeline;

id<MTLComputePipelineState> g_unary_sqrt_pipeline;

id<MTLComputePipelineState> g_unary_clamp_pipeline;

id<MTLComputePipelineState> g_unary_scale_pipeline;

id<MTLComputePipelineState> g_unary_fill_pipeline;

id<MTLComputePipelineState> g_unary_fill_f16_pipeline;

id<MTLComputePipelineState> g_bin_mul_scalar_pipeline;

id<MTLComputePipelineState> g_bin_div_row_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_sum6_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;

id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_sum6_pipeline;

id<MTLComputePipelineState> g_rope_tail_batch_pipeline;

id<MTLComputePipelineState> g_dsv4_fp8_kv_quantize_pipeline;

id<MTLComputePipelineState> g_dsv4_indexer_qat_pipeline;

id<MTLComputePipelineState> g_dsv4_kv_fp8_store_pipeline;

id<MTLComputePipelineState> g_dsv4_ratio4_shift_pipeline;

id<MTLComputePipelineState> g_dsv4_softmax_pool_pipeline;

id<MTLComputePipelineState> g_soft_max_f32_pipeline;

id<MTLComputePipelineState> g_soft_max_f32_4_pipeline;

id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;

id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;

id<MTLComputePipelineState> g_sum_rows_f32_f32_pipeline;

id<MTLComputePipelineState> g_dsv4_topk_mask_pipeline;

id<MTLComputePipelineState> g_dsv4_topk_mask_scatter_pipeline;

id<MTLComputePipelineState> g_dsv4_indexer_weighted_sum_pipeline;

id<MTLComputePipelineState> g_dsv4_indexer_score_one_direct_pipeline;

id<MTLComputePipelineState> g_dsv4_compressor_store_one_pipeline;

id<MTLComputePipelineState> g_dsv4_sort_i32_rows_asc_pipeline;

id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_pipeline;

id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_rb16_pipeline;

id<MTLComputePipelineState> g_dsv4_softplus_sqrt_pipeline;

id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline;

id<MTLComputePipelineState> g_dsv4_router_weights_one_pipeline;

id<MTLComputePipelineState> g_dsv4_route_translate_pipeline;

id<MTLComputePipelineState> g_dsv4_hc_expand4_pipeline;

NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;

NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;

NSMutableArray<id<MTLBuffer>> *g_transient_buffers;

id g_model_residency_set;

id<MTLBuffer> g_flash_attn_mask_buffer;

id<MTLBuffer> g_flash_attn_pad_buffer;

id<MTLBuffer> g_flash_attn_tmp_buffer;

id<MTLBuffer> g_flash_attn_blk_buffer;

id<MTLBuffer> g_flash_attn_ring_buffer;

id<MTLBuffer> g_flash_attn_kv_buffer;

id<MTLBuffer> g_compressor_pool_kv_buffer;

id<MTLBuffer> g_compressor_pool_score_buffer;

id<MTLBuffer> g_compressor_pool_score_cont_buffer;

id<MTLBuffer> g_compressor_pool_softmax_buffer;

id<MTLBuffer> g_compressor_pool_product_buffer;

id<MTLBuffer> g_compressor_store_ape_buffer;

id<MTLBuffer> g_compressor_store_score_buffer;

id<MTLBuffer> g_embed_rows_buffer;

id<MTLBuffer> g_router_selection_buffer;

id<MTLBuffer> g_router_weight_sum_buffer;

/* Resident original-id -> compact-slot LUT for reduced-expert models, n_layer*256
 * int16. nil for a full model (translation kernel never dispatched). */
id<MTLBuffer> g_expert_keep_lut_buffer;

uint32_t g_expert_keep_lut_layers;

/* B2b verify-mode clamp tally (react-go-execution-plan M-1.2): [n_layer*256]
 * uint32, atomic-incremented by kernel_dsv4_route_translate whenever a routed
 * expert id is clamped to slot 0 (i.e. a cold/dropped expert leaked into the
 * top-k). Persists across teardown like the REAP buffers so its atexit dump
 * still sees the last forward's result. nil until DS4_VERIFY_ROUTE_CLAMP. */
int           g_route_clamp_verify = -1;

/* -1 uninit, 0 off, 1 on */
id<MTLBuffer> g_route_clamp_buf;

uint32_t      g_route_clamp_layers;

/* layer count the buffer is sized for */
id<MTLBuffer> g_indexer_head_scores_buffer;

id<MTLBuffer> g_indexer_topk_buffer;

id<MTLBuffer> g_indexed_topk_buffer;

id<MTLBuffer> g_f16_round_scratch_buffer;

id<MTLBuffer> g_raw_store_round_buffer;

id<MTLBuffer> g_moe_gate_scratch_buffer;

id<MTLBuffer> g_moe_down_scratch_buffer;

id<MTLBuffer> g_moe_id_map_buffer;

id<MTLBuffer> g_attn_out_group_ids_buffer;

/* A3 routed-expert offload scratch.  When DS4_METAL_EXPERT_OFFLOAD=1 the
 * routed MoE kernels read only the active expert slots copied from the GGUF mmap
 * into these compact resident buffers, instead of binding the huge mmap-backed
 * expert tensors to each command buffer. */
id<MTLBuffer> g_moe_scratch_gate;

id<MTLBuffer> g_moe_scratch_up;

id<MTLBuffer> g_moe_scratch_down;

id<MTLBuffer> g_moe_go1b_res_scratch;

/* 1-bit residual mapped-tile matmul output */
id<MTLBuffer> g_moe_res_gate_scratch;

/* compacted residual gate experts (n_active) */
id<MTLBuffer> g_moe_res_up_scratch;

/* compacted residual up experts (n_active) */
id<MTLBuffer> g_moe_res_down_scratch;

/* compacted residual down experts (n_active) */
/* R5-C go2b hot/cold split: hot picks run one go2b matmul pass; base pass masks them. */
id<MTLBuffer> g_moe_hot_gate_scratch;

id<MTLBuffer> g_moe_hot_up_scratch;

id<MTLBuffer> g_moe_hot_down_scratch;

id<MTLBuffer> g_moe_hot_sel;

/* per-pick compact hot slot (i32) or 0xFFFF */
int32_t *g_hot_pick_slot;

/* host: per-pick compact hot index or -1 */
uint32_t g_hot_pick_cap;

/* go1b corr: snapshot of the ORIGINAL top-k expert ids taken inside the offload
 * MoE before it remaps the selected buffer to compact slots in place.  The corr
 * (ds4_gpu_corr_apply) indexes per-expert C[e]/beta[e] by true id, so it must
 * read this pre-remap copy, not the corrupted live selected tensor. */
ds4_gpu_tensor *g_corr_saved_selected;

ds4_gpu_tensor *ds4_gpu_corr_saved_selected(void) { return g_corr_saved_selected; }

/* Ensure the compacted residual weight scratches (Shared: CPU-gathered, GPU-read). */
int ds4_gpu_ensure_res_scratch(uint32_t n_active, uint64_t expert_bytes) {
    uint64_t need = (uint64_t)n_active * expert_bytes;
    if (need == 0) return 0;
    if (!g_moe_res_gate_scratch || (uint64_t)g_moe_res_gate_scratch.length < need)
        g_moe_res_gate_scratch = [g_device newBufferWithLength:need options:MTLResourceStorageModeShared];
    if (!g_moe_res_up_scratch || (uint64_t)g_moe_res_up_scratch.length < need)
        g_moe_res_up_scratch = [g_device newBufferWithLength:need options:MTLResourceStorageModeShared];
    return g_moe_res_gate_scratch != nil && g_moe_res_up_scratch != nil;
}

const void *g_model_map_ptr;

uint64_t g_model_map_size;

/* Largest model file size ever mapped this process. The expert-fetch transport
 * reads expert bytes from the BASE GGUF (~81 GiB) and handshakes the peer's
 * served file size against it; but loading the small MTP draft model
 * (~2.14 GiB, 本机 MTP topology) overwrites g_model_map_size with the draft's
 * size, which made the efetch handshake reject every connection ("remote size
 * <base> vs local <draft>") and silently starved the coordinator of the peer's
 * fast SSD. Tracking the max (the base model is always the largest) keeps the
 * efetch size stable regardless of MTP load order. */
uint64_t g_efetch_model_size;

uint64_t g_model_mapped_offset;

uint64_t g_model_mapped_size;

uint64_t g_model_mapped_max_tensor_bytes;

uint64_t g_tensor_alloc_live_bytes;

uint64_t g_tensor_alloc_peak_bytes;

uint64_t g_model_wrap_count;

uint64_t g_model_wrap_bytes;

uint64_t g_model_wrap_max_bytes;

uint64_t g_model_residency_count;

int g_metal4_runtime_available;

int g_metal4_family_supported;

int g_metal4_queue_supported;

int g_metal4_m5_neural_accelerators_hint;

int g_metal4_tensor_api_enabled;

int g_metal4_tensor_api_compile_supported;

char g_metal_device_name[128];

NSUInteger g_flash_attn_mask_bytes;

NSUInteger g_flash_attn_pad_bytes;

NSUInteger g_flash_attn_tmp_bytes;

NSUInteger g_flash_attn_blk_bytes;

NSUInteger g_flash_attn_ring_bytes;

NSUInteger g_flash_attn_kv_bytes;

NSUInteger g_compressor_pool_kv_bytes;

NSUInteger g_compressor_pool_score_bytes;

NSUInteger g_compressor_pool_score_cont_bytes;

NSUInteger g_compressor_pool_softmax_bytes;

NSUInteger g_compressor_pool_product_bytes;

NSUInteger g_compressor_store_ape_bytes;

NSUInteger g_compressor_store_score_bytes;

NSUInteger g_embed_rows_bytes;

NSUInteger g_router_selection_bytes;

NSUInteger g_router_weight_sum_bytes;

NSUInteger g_indexer_head_scores_bytes;

NSUInteger g_indexer_topk_bytes;

NSUInteger g_indexed_topk_bytes;

NSUInteger g_f16_round_scratch_bytes;

NSUInteger g_raw_store_round_bytes;

NSUInteger g_moe_gate_scratch_bytes;

NSUInteger g_moe_down_scratch_bytes;

NSUInteger g_moe_id_map_bytes;

NSUInteger g_attn_out_group_ids_bytes;

NSUInteger g_moe_scratch_gate_bytes;

NSUInteger g_moe_scratch_up_bytes;

NSUInteger g_moe_scratch_down_bytes;

int g_initialized;

int g_quality_mode;

int g_mpp_invalid_env_reported;

static uint64_t ds4_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

void ds4_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = ds4_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "ds4: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "ds4: Metal device %s\n", name);
    }
}

ds4_gpu_model_view g_model_views[DS4_METAL_MAX_MODEL_VIEWS];

uint32_t g_model_view_count;

@implementation DS4MetalTensor
@end

/* When set, the next contiguous-range model map (ds4_gpu_set_model_map_range)
 * wraps its views WITHOUT adding them to the GPU residency set, so their clean
 * mmap pages stay reclaimable instead of pinning the wired working set.  Used
 * for the MTP draft model on the memory-tight worker (DS4_MTP_NO_RESIDENCY):
 * the draft tensors are read via the no-copy mmap views (fine on Metal) and the
 * hot, every-step ones stay warm in the page cache.  Auto-resets after use. */
int g_model_view_force_nonresident;
