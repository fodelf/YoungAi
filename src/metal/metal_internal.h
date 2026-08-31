/* metal_internal.h — src/metal 全组 .m 的内部共享头(机械搬移自 ds4_metal.m)。
 * 只被 src/metal 下的 .m 包含: ds4_gpu_model_view/DS4MetalTensor 带 __strong ObjC 成员, 不能进任何 .c。 */
#ifndef DS4_METAL_INTERNAL_H
#define DS4_METAL_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <Accelerate/Accelerate.h>
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
    DS4_METAL_TENSOR_F16W    = 1,    /* v2.2 VQ gather 的 f16 权重 scratch(半精度直读) */
    DS4_METAL_TENSOR_Q2_K    = 10,
    DS4_METAL_TENSOR_Q4_K    = 12,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
    DS4_METAL_TENSOR_GO1B    = 40,   /* strict-1-bit routed expert (mirrors GGUF ggml type 40) */
    DS4_METAL_TENSOR_GO2B    = 41,   /* merged base+residual binary pair (R5-C go2b) */
};

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
     * memory pressure. Used for routed-expert weights under expert offload;
     * every view is resident by default (full-model behaviour unchanged). */
    bool resident_hint;
} ds4_gpu_model_view;

@interface DS4MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
@end

enum {
    DS4_METAL_ATTN_OUT_MPP_TILE_N = 64,
    /* Small-batch matvec-ext kernel cap: above this token count the mm/tile
     * paths win. Production value; the mv-ext variants only exist for <=8. */
    DS4_METAL_SMALL_BATCH_MV_MAX_TOKENS = 8,
    /* FlashAttention prefill: token count at which the long (nonvec) variant
     * beats the vec variant. */
    DS4_METAL_FA_LONG_MIN_TOKENS = 20,
    /* Attention-output projection: matrix (mm_id) path from this batch size,
     * vector kernels below it. */
    DS4_METAL_ATTN_OUT_MM_MIN_TOKENS = 32,
    /* Routed MoE grouped-GEMM threshold: mm_id maps token-rows per expert and
     * reads each active expert's weights exactly once; below it the pair/mv
     * kernels win. 8 covers the 5..12-token verify shapes (measured). */
    DS4_METAL_MOE_MM_ID_MIN_TOKENS = 8,
    /* Below the GEMM threshold, batches up to this size take the pair kernels
     * (one activation read computes routed gate/up together per pick). */
    DS4_METAL_MOE_TINY_PAIR_MAX_TOKENS = 16,
    /* Expert gather worker threads. 生产值: svc.sh/双机 lane 一贯传 8; admit 路
     * 旧默认 1 是从未被脚本采用的陈值 —— 写死 8 是刻意的默认对齐, 不是笔误。 */
    DS4_METAL_EXPERT_GATHER_THREADS = 8,
    /* Batch gathers at/above this token count read through a separate
     * F_NOCACHE fd (one-shot prefill bytes must not evict hot pages); the
     * K=64 ladder sends kc=25/33 verify batches whose unions ARE warm, so 64
     * keeps every verify batch (protocol max 64 rows) on the cached fd. */
    DS4_METAL_BATCH_NOCACHE_MIN_TOKENS = 64,
    /* Host wait ceiling for the TP rendezvous AND the A3 event drain: both
     * MTLSharedEvent waits time out to the same standard before falling back
     * to waitUntilCompleted. */
    DS4_METAL_TP_EVENT_TIMEOUT_MS = 60000,
    /* Expert-fetch dial retries: every 2s for up to 150 attempts (~5 minutes,
     * long enough to reach the decode phase where the link is idle); after
     * that remote fetch is permanently disabled for the run. */
    DS4_METAL_EFETCH_CONNECT_ATTEMPTS_MAX = 150,
    /* Post-accept kick thread: 10 dial tries spaced 700ms cover the quiet
     * window right after the control connection is established. */
    DS4_METAL_EFETCH_KICK_TRIES = 10,
    DS4_METAL_EFETCH_KICK_INTERVAL_US = 700000,
    /* DeepSeek V4 routes 6 experts per token; pool bookkeeping floors. */
    DS4_METAL_ROUTED_TOPK = 6,
    /* One MoE layer call can activate at most this many distinct experts
     * (compact scratch slot id space; also the collect/remap LUT size). */
    DS4_METAL_ACTIVE_EXPERTS_MAX = 1024,
};

/* VQ 层全专家 dequant f16 scratch 的护栏上限(2*gate/up + down 合计)。 */
#define DS4_METAL_VQ_SCRATCH_CAP_BYTES (3ull * 1024ull * 1024ull * 1024ull)

#include "metal_args.h"

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;
extern id<MTLLibrary> g_library;
extern id<MTLCommandBuffer> g_batch_cb;
extern id<MTLComputeCommandEncoder> g_batch_enc;
extern NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
extern id<MTLComputePipelineState> g_set_rows_f32_i32_pipeline, g_get_rows_f32_pipeline, g_get_rows_f16_pipeline, g_get_rows_i32_pipeline;
extern id<MTLComputePipelineState> g_repeat_f32_pipeline, g_concat_pipeline, g_cpy_f32_f32_pipeline, g_cpy_f32_f16_pipeline;
extern id<MTLComputePipelineState> g_cpy_f16_f32_pipeline, g_swiglu_pipeline, g_add_pipeline, g_moe_sum6_pipeline;
extern id<MTLComputePipelineState> g_mul_pipeline, g_rms_norm_pipeline, g_rms_norm_plain_pipeline, g_dsv4_qkv_rms_norm_pipeline;
extern id<MTLComputePipelineState> g_hc_split_sinkhorn_pipeline, g_hc_split_weighted_sum_pipeline, g_hc_split_weighted_sum_norm_pipeline, g_hc_weighted_sum_pipeline;
extern id<MTLComputePipelineState> g_hc_expand_pipeline, g_unary_sigmoid_pipeline, g_unary_silu_pipeline, g_unary_softplus_pipeline;
extern id<MTLComputePipelineState> g_unary_sqrt_pipeline, g_unary_clamp_pipeline, g_unary_scale_pipeline, g_unary_fill_pipeline;
extern id<MTLComputePipelineState> g_unary_fill_f16_pipeline, g_bin_mul_scalar_pipeline, g_bin_div_row_pipeline, g_moe_mul_mv_id_iq2_xxs_pipeline;
extern id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_pipeline, g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline, g_moe_mul_mv_id_q2_k_pipeline, g_moe_mul_mv_id_q2_k_sum6_pipeline;
extern id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pipeline, g_moe_mul_mv_id_q4_k_pair_pipeline, g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline, g_moe_mul_mv_id_q4_k_sum6_pipeline;
extern id<MTLComputePipelineState> g_rope_tail_batch_pipeline, g_dsv4_fp8_kv_quantize_pipeline, g_dsv4_indexer_qat_pipeline, g_dsv4_kv_fp8_store_pipeline;
extern id<MTLComputePipelineState> g_dsv4_ratio4_shift_pipeline, g_dsv4_softmax_pool_pipeline, g_soft_max_f32_pipeline, g_soft_max_f32_4_pipeline;
extern id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline, g_argsort_merge_f32_i32_desc_pipeline, g_sum_rows_f32_f32_pipeline, g_dsv4_topk_mask_pipeline;
extern id<MTLComputePipelineState> g_dsv4_topk_mask_scatter_pipeline, g_dsv4_indexer_weighted_sum_pipeline, g_dsv4_indexer_score_one_direct_pipeline, g_dsv4_compressor_store_one_pipeline;
extern id<MTLComputePipelineState> g_dsv4_sort_i32_rows_asc_pipeline, g_dsv4_indexed_attention_heads8_pipeline, g_dsv4_indexed_attention_heads8_rb16_pipeline, g_dsv4_softplus_sqrt_pipeline;
extern id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline, g_dsv4_router_weights_one_pipeline, g_dsv4_route_translate_pipeline, g_dsv4_hc_expand4_pipeline;
extern NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;
extern NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;
extern NSMutableArray<id<MTLBuffer>> *g_transient_buffers;
extern id g_model_residency_set;
extern id<MTLBuffer> g_flash_attn_mask_buffer, g_flash_attn_pad_buffer, g_flash_attn_tmp_buffer, g_flash_attn_blk_buffer;
extern id<MTLBuffer> g_flash_attn_ring_buffer, g_flash_attn_kv_buffer, g_compressor_pool_kv_buffer, g_compressor_pool_score_buffer;
extern id<MTLBuffer> g_compressor_pool_score_cont_buffer, g_compressor_pool_softmax_buffer, g_compressor_pool_product_buffer, g_compressor_store_ape_buffer;
extern id<MTLBuffer> g_compressor_store_score_buffer, g_embed_rows_buffer, g_router_selection_buffer, g_router_weight_sum_buffer;
extern id<MTLBuffer> g_expert_keep_lut_buffer;
extern uint32_t g_expert_keep_lut_layers;
extern id<MTLBuffer> g_route_clamp_buf;
extern uint32_t g_route_clamp_layers;
extern id<MTLBuffer> g_indexer_head_scores_buffer, g_indexer_topk_buffer, g_indexed_topk_buffer, g_f16_round_scratch_buffer;
extern id<MTLBuffer> g_raw_store_round_buffer, g_moe_gate_scratch_buffer, g_moe_down_scratch_buffer, g_moe_id_map_buffer;
extern id<MTLBuffer> g_attn_out_group_ids_buffer, g_moe_scratch_gate, g_moe_scratch_up, g_moe_scratch_down;
extern id<MTLBuffer> g_moe_go1b_res_scratch, g_moe_res_gate_scratch, g_moe_res_up_scratch, g_moe_res_down_scratch;
extern id<MTLBuffer> g_moe_hot_gate_scratch, g_moe_hot_up_scratch, g_moe_hot_down_scratch, g_moe_hot_sel;
extern int32_t *g_hot_pick_slot;
extern uint32_t g_hot_pick_cap;
extern ds4_gpu_tensor *g_corr_saved_selected;
extern const void *g_model_map_ptr;
extern uint64_t g_model_map_size, g_efetch_model_size, g_model_mapped_offset, g_model_mapped_size;
extern uint64_t g_model_mapped_max_tensor_bytes, g_tensor_alloc_live_bytes, g_tensor_alloc_peak_bytes, g_model_wrap_count;
extern uint64_t g_model_wrap_bytes, g_model_wrap_max_bytes, g_model_residency_count;
extern int g_metal4_runtime_available, g_metal4_family_supported, g_metal4_queue_supported, g_metal4_m5_neural_accelerators_hint;
extern int g_metal4_tensor_api_enabled, g_metal4_tensor_api_compile_supported;
extern char g_metal_device_name[128];
extern NSUInteger g_flash_attn_mask_bytes, g_flash_attn_pad_bytes, g_flash_attn_tmp_bytes, g_flash_attn_blk_bytes;
extern NSUInteger g_flash_attn_ring_bytes, g_flash_attn_kv_bytes, g_compressor_pool_kv_bytes, g_compressor_pool_score_bytes;
extern NSUInteger g_compressor_pool_score_cont_bytes, g_compressor_pool_softmax_bytes, g_compressor_pool_product_bytes, g_compressor_store_ape_bytes;
extern NSUInteger g_compressor_store_score_bytes, g_embed_rows_bytes, g_router_selection_bytes, g_router_weight_sum_bytes;
extern NSUInteger g_indexer_head_scores_bytes, g_indexer_topk_bytes, g_indexed_topk_bytes, g_f16_round_scratch_bytes;
extern NSUInteger g_raw_store_round_bytes, g_moe_gate_scratch_bytes, g_moe_down_scratch_bytes, g_moe_id_map_bytes;
extern NSUInteger g_attn_out_group_ids_bytes, g_moe_scratch_gate_bytes, g_moe_scratch_up_bytes, g_moe_scratch_down_bytes;
extern int g_initialized, g_quality_mode, g_strict_fp, g_metal4_enabled;
extern ds4_gpu_model_view g_model_views[DS4_METAL_MAX_MODEL_VIEWS];
extern uint32_t g_model_view_count;
extern int g_model_fd, g_model_fd_conflict;
extern int g_expert_offload_verdict;

int ds4_gpu_ensure_res_scratch(uint32_t n_active, uint64_t expert_bytes);
void ds4_gpu_print_device_summary(void);
void ds4_gpu_close_batch_encoder(void);
int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label);
int ds4_gpu_wait_pending_command_buffers(const char *label);
int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label);
int ds4_gpu_submit_command_buffer_async(id<MTLCommandBuffer> cb, int owned);
int ds4_gpu_ensure_scratch_buffer( id<MTLBuffer> __strong *buffer, NSUInteger *capacity, NSUInteger bytes, const char *label);
uint64_t ds4_gpu_effective_model_max_tensor_bytes(uint64_t map_size, uint64_t max_tensor_bytes);
double ds4_gpu_now_ms(void);
void ds4_gpu_progress_begin(const char *what);
void ds4_gpu_progress_done(void);
void ds4_gpu_progress_failed(void);
void ds4_gpu_model_views_clear(void);
void ds4_gpu_model_residency_clear(void);
int ds4_gpu_add_model_view_range( const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes, bool resident, uint64_t *mapped_model_size_out);
int ds4_gpu_finish_model_views( double t0, uint64_t mapped_model_size, uint64_t display_offset);
int ds4_gpu_map_model_views( const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
id<MTLBuffer> ds4_gpu_new_transient_buffer(NSUInteger bytes, const char *label);
id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline( const char *function_name, bool bc_inp, bool bc_out);
id<MTLComputePipelineState> ds4_gpu_get_mul_mm_id_pipeline( const char *function_name, bool bc_inp);
id<MTLComputePipelineState> ds4_gpu_get_pipeline( const char *function_name);
id<MTLComputePipelineState> ds4_gpu_hot_pipeline( id<MTLComputePipelineState> pipeline, const char *fallback_name);
int ds4_gpu_mpp_available(void);
int ds4_gpu_use_mpp_attn_out_low_matmul(void);
void ds4_gpu_warn_mpp_fallback(void);
int ds4_gpu_device_name_contains(const char *needle);
void ds4_gpu_detect_metal4_features(void);
int ds4_gpu_warm_model_views(void);
const char *ds4_gpu_mul_mm_id_map0_name(uint32_t ne20);
id<MTLComputePipelineState> ds4_gpu_get_mul_mv_pipeline( const char *function_name, int16_t nsg);
id<MTLComputePipelineState> ds4_gpu_get_mul_mv_ext_pipeline( const char *function_name, int16_t nsg, int16_t nxpsg);
id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pad_pipeline( bool has_mask, int32_t ncpsg);
id<MTLComputePipelineState> ds4_gpu_get_flash_attn_blk_pipeline( int32_t nqptg, int32_t ncpsg);
id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pipeline( const char *function_name, bool has_mask, bool has_sinks, bool has_bias, bool has_scap, bool has_kvpad, bool bc_mask, int32_t ns10, int32_t ns20, int32_t nsg);
id<MTLComputePipelineState> ds4_gpu_get_flash_attn_vec_pipeline( const char *function_name, bool has_mask, bool has_sinks, bool has_bias, bool has_scap, bool has_kvpad, int32_t ns10, int32_t ns20, int32_t nsg, int32_t nwg);
id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_pipeline( int32_t dv, int32_t nwg);
uint32_t ds4_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg);
NSString *ds4_gpu_full_source(void);
ds4_gpu_cpy_args ds4_gpu_make_cpy_1d_args( uint32_t n, uint64_t src_elem, uint64_t dst_elem);
ds4_gpu_bin_args ds4_gpu_make_bin_rows_args(uint32_t n, uint32_t rows, uint32_t rhs_n);
ds4_gpu_unary_args ds4_gpu_make_unary_rows_args( uint32_t n, uint32_t rows, int c4, float scale, float bias);
ds4_gpu_bin_args ds4_gpu_make_bin_same_rows_args(uint32_t n, uint32_t rows);
ds4_gpu_q8_0_matvec_args ds4_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim);
ds4_gpu_f16_matvec_args ds4_gpu_make_f16_mv_args(uint64_t in_dim, uint64_t out_dim);
ds4_gpu_q8_0_matvec_args ds4_gpu_make_f32_mv_args( uint64_t in_dim, uint64_t out_dim, uint64_t n_vec);
ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_mv_dispatch(void);
ds4_gpu_mv_dispatch ds4_gpu_make_plain_mv_dispatch( uint64_t in_dim, int f32_weights);
ds4_gpu_mul_mm_args ds4_gpu_make_mm_args( uint64_t in_dim, uint64_t out_dim, uint64_t n_tok, uint64_t row_bytes);
ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args( uint64_t in_dim, uint64_t out_dim, uint64_t n_tok, uint64_t elem_bytes, uint64_t row_bytes);
int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok);
int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok);
const char *ds4_gpu_mv_ext_name(int q8, int16_t r1ptg);
ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps);
ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_3d_args( uint32_t n0, uint32_t n1, uint32_t n2, float eps);
NSUInteger ds4_gpu_rms_norm_threads(uint32_t n);
NSUInteger ds4_gpu_rms_norm_pipeline_threads( uint32_t n, id<MTLComputePipelineState> pipeline);
ds4_gpu_rope_tail_batch_args ds4_gpu_make_rope_tail_args( uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
int ds4_gpu_encode_rope_tail_inplace( id<MTLCommandBuffer> cb, id<MTLBuffer> xbuf, NSUInteger xoff, const ds4_gpu_rope_tail_batch_args *args, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t pos0, uint32_t pos_step);
int ds4_gpu_init(void);
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
int ds4_gpu_expert_drain_commands(const char *label);
int ds4_gpu_encode_get_rows_f16( id<MTLCommandBuffer> cb, id<MTLBuffer> weight, NSUInteger weight_offset, id<MTLBuffer> tokens, NSUInteger tokens_offset, id<MTLBuffer> out, NSUInteger out_offset, uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd);
int ds4_gpu_encode_repeat_hc_embedding( id<MTLCommandBuffer> cb, id<MTLBuffer> rows, NSUInteger rows_offset, id<MTLBuffer> out, NSUInteger out_offset, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
id<MTLBuffer> ds4_gpu_wrap_model_range( const void *model_map, uint64_t model_size, uint64_t offset, uint64_t len, uint64_t *inner_offset);
int ds4_gpu_indexer_topk_tensor( ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k);
int ds4_gpu_matmul_q8_0_tensor( ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_rms_norm_weight_rows_tensor( ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps);
int ds4_gpu_head_rms_norm_tensor( ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps);
int ds4_gpu_rope_tail_tensor( ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
int ds4_gpu_dsv4_fp8_kv_quantize_tensor( ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot);
int ds4_gpu_encode_set_rows_f32_i32( id<MTLCommandBuffer> cb, ds4_gpu_tensor *dst, id<MTLBuffer> srcbuf, NSUInteger src_off, const int32_t *rows, uint32_t n_rows, uint32_t dst_rows, uint32_t width);
int ds4_gpu_encode_add_f32_1d( id<MTLCommandBuffer> cb, id<MTLBuffer> a, NSUInteger a_off, id<MTLBuffer> b, NSUInteger b_off, id<MTLBuffer> out, NSUInteger out_off, uint32_t n);
int ds4_gpu_kv_fp8_store_raw_tensor( ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t row, uint32_t head_dim, uint32_t n_rot);
int ds4_gpu_encode_compressor_score_with_ape( id<MTLCommandBuffer> cb, id<MTLBuffer> score_src, NSUInteger score_src_offset, id<MTLBuffer> score_dst, NSUInteger score_dst_offset, id<MTLBuffer> apebuf, NSUInteger ape_offset, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t n_tokens);
int ds4_gpu_encode_compressor_set_rows_projected( id<MTLCommandBuffer> cb, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, id<MTLBuffer> kvbuf, NSUInteger kv_offset, id<MTLBuffer> scorebuf, NSUInteger score_offset, id<MTLBuffer> apebuf, NSUInteger ape_offset, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0, const int32_t *rows, uint32_t n_rows, uint32_t state_rows);
int ds4_gpu_compressor_store_one_tensor( const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos);
int ds4_gpu_encode_dsv4_softmax_pool( id<MTLCommandBuffer> cb, ds4_gpu_tensor *out, id<MTLBuffer> kvbuf, NSUInteger kv_offset, uint64_t kv_nb0, uint64_t kv_nb1, uint64_t kv_nb2, id<MTLBuffer> scorebuf, NSUInteger score_offset, uint64_t score_nb0, uint64_t score_nb1, uint64_t score_nb2, uint32_t n_rows, uint32_t head_dim, uint32_t n_comp);
int ds4_gpu_encode_compressor_pool( id<MTLCommandBuffer> cb, ds4_gpu_tensor *out, const ds4_gpu_tensor *state_kv, const ds4_gpu_tensor *state_score, uint32_t head_dim, uint32_t ratio);
int ds4_gpu_encode_compressor_shift_ratio4( id<MTLCommandBuffer> cb, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, uint32_t width);
int ds4_gpu_encode_fill_f32_rows( id<MTLCommandBuffer> cb, id<MTLBuffer> buf, NSUInteger offset, uint32_t width, uint32_t rows, float value);
int ds4_gpu_encode_cpy_f32_f32_1d( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t n);
int ds4_gpu_encode_cpy_f32_f32_3d( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t cols, uint32_t rows, uint32_t planes, uint64_t src_row_stride, uint64_t src_plane_stride, uint64_t dst_row_stride, uint64_t dst_plane_stride);
int ds4_gpu_encode_cpy_f32_f32_3d_src_strided( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t cols, uint32_t rows, uint32_t planes, uint64_t src_col_stride, uint64_t src_row_stride, uint64_t src_plane_stride, uint64_t dst_row_stride, uint64_t dst_plane_stride);
int ds4_gpu_encode_cpy_f32_f16_1d( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t n);
int ds4_gpu_encode_cpy_f32_f16_2d( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t cols, uint32_t rows, uint64_t src_row_stride, uint64_t dst_row_stride);
int ds4_gpu_encode_cpy_f16_f32_1d( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t n);
int ds4_gpu_encode_copy_to_f16_1d( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, bool src_is_f16, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t n);
int ds4_gpu_encode_fill_f16_1d( id<MTLCommandBuffer> cb, id<MTLBuffer> buf, NSUInteger offset, uint32_t n, float value);
int ds4_gpu_encode_flash_attention_raw_heads( id<MTLCommandBuffer> cb, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_head, uint32_t head_dim);
void ds4_gpu_fill_raw_prefill_mask(uint16_t *mask, uint32_t n_tokens, uint32_t window);
void ds4_gpu_fill_raw_decode_batch_mask( uint16_t *mask, uint32_t n_tokens, uint32_t n_raw, uint32_t pos0, uint32_t window);
void ds4_gpu_fill_mixed_decode_batch_mask( uint16_t *mask, uint32_t n_tokens, uint32_t n_raw, uint32_t n_comp, uint32_t pos0, uint32_t window, uint32_t ratio);
void ds4_gpu_fill_static_mixed_prefill_mask( uint16_t *mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio);
int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long( id<MTLCommandBuffer> __strong *cbp, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec( id<MTLCommandBuffer> __strong *cbp, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_encode_flash_attention_prefill_raw_heads( id<MTLCommandBuffer> __strong *cbp, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_encode_flash_attention_gathered_heads( id<MTLCommandBuffer> cb, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, uint32_t n_comp, const ds4_gpu_tensor *comp_mask, uint32_t use_mask, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_encode_flash_attention_decode_raw_batch_heads( id<MTLCommandBuffer> cb, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t window, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_encode_flash_attention_decode_mixed_batch_heads( id<MTLCommandBuffer> cb, ds4_gpu_tensor *heads, id<MTLBuffer> sinks_buf, NSUInteger sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_corr_apply( ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const ds4_gpu_tensor *U, const ds4_gpu_tensor *V, const ds4_gpu_tensor *C, const ds4_gpu_tensor *b, const ds4_gpu_tensor *beta, const ds4_gpu_tensor *selected, uint32_t d_model, uint32_t d_l, uint32_t n_expert, uint32_t n_expert_used, uint32_t n_tokens);
int ds4_gpu_encode_unary_f32_rows( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t width, uint32_t rows, int c4, float min, float max);
int ds4_gpu_encode_bin_f32_rows( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_bin_args *args, id<MTLBuffer> a, NSUInteger a_off, id<MTLBuffer> b, NSUInteger b_off, id<MTLBuffer> out, NSUInteger out_off);
ds4_gpu_bin_args ds4_gpu_make_bin_rowwise_scalar_args(uint32_t width, uint32_t rows);
ds4_gpu_mul_mv_id_args ds4_gpu_make_mul_mv_id_args( uint32_t src0_cols, uint32_t src0_rows, uint32_t src0_experts, uint64_t src0_row_bytes, uint64_t src0_expert_bytes, uint32_t src1_expert_rows, uint32_t selected_experts, uint32_t n_tokens, uint32_t nr0);
ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args( uint32_t src0_cols, uint32_t src0_experts, uint32_t src1_expert_rows, uint32_t selected_experts, uint32_t n_tokens);
ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args( uint32_t src0_cols, uint32_t src0_rows, uint32_t src0_experts, uint64_t src0_row_bytes, uint64_t src0_expert_bytes, uint32_t src1_expert_rows, uint32_t selected_experts, uint32_t n_tokens);
ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size( uint32_t src0_cols, uint32_t src0_rows, uint32_t src0_experts, uint64_t src0_row_bytes, uint64_t src0_expert_bytes, uint32_t src1_expert_rows, uint32_t selected_experts, uint32_t n_tokens, uint32_t src1_elem_size);
uint32_t ds4_gpu_routed_mv_nr0(uint32_t type);
const char *ds4_gpu_metal_tensor_type_name(uint32_t type);
NSUInteger ds4_gpu_routed_mv_smem(uint32_t type);
id<MTLComputePipelineState> ds4_gpu_routed_mv_pipeline(uint32_t type);
id<MTLComputePipelineState> ds4_gpu_routed_mm_pipeline(uint32_t type);
id<MTLComputePipelineState> ds4_gpu_routed_mm_f16_rhs_pipeline(uint32_t type);
int ds4_gpu_encode_mul_mv_id( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_mul_mv_id_args *args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off, id<MTLBuffer> ids, NSUInteger ids_off, NSUInteger threadgroup_bytes, NSUInteger nsg, bool rows_per_group_is_nr0);
int ds4_gpu_encode_attn_out_low_q8_direct( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_mul_mv_id_args *args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off, NSUInteger threadgroup_bytes, NSUInteger nsg);
int ds4_gpu_encode_mul_mv_id_pair( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_mul_mv_id_args *args, id<MTLBuffer> src0_a, NSUInteger src0_a_off, id<MTLBuffer> src0_b, NSUInteger src0_b_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst_a, NSUInteger dst_a_off, id<MTLBuffer> dst_b, NSUInteger dst_b_off, id<MTLBuffer> ids, NSUInteger ids_off, NSUInteger threadgroup_bytes, NSUInteger nsg, bool rows_per_group_is_nr0);
int ds4_gpu_encode_mul_mv_id_pair_swiglu( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_mul_mv_id_args *args, const ds4_gpu_dsv4_moe_swiglu_weight_args *act, id<MTLBuffer> src0_a, NSUInteger src0_a_off, id<MTLBuffer> src0_b, NSUInteger src0_b_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst_a, NSUInteger dst_a_off, id<MTLBuffer> dst_b, NSUInteger dst_b_off, id<MTLBuffer> dst_mid, NSUInteger dst_mid_off, id<MTLBuffer> ids, NSUInteger ids_off, id<MTLBuffer> weights, NSUInteger weights_off, NSUInteger threadgroup_bytes, NSUInteger nsg, bool rows_per_group_is_nr0);
int ds4_gpu_encode_mul_mv_id_sum6( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_mul_mv_id_args *args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off, id<MTLBuffer> ids, NSUInteger ids_off, NSUInteger threadgroup_bytes, NSUInteger nsg);
int ds4_gpu_encode_mul_mm_id( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> map_pipeline, id<MTLComputePipelineState> mm_pipeline, const ds4_gpu_mul_mm_id_map_args *map_args, const ds4_gpu_mul_mm_id_args *mm_args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off, id<MTLBuffer> ids, NSUInteger ids_off);
int ds4_gpu_encode_mul_mm_id_map( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> map_pipeline, const ds4_gpu_mul_mm_id_map_args *map_args, const ds4_gpu_mul_mm_id_args *mm_args, id<MTLBuffer> ids, NSUInteger ids_off);
int ds4_gpu_encode_mul_mm_id_mapped_tile( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> mm_pipeline, const ds4_gpu_mul_mm_id_args *mm_args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off);
int ds4_gpu_encode_mul_mm_id_mapped( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> mm_pipeline, const ds4_gpu_mul_mm_id_args *mm_args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off);
int ds4_gpu_encode_attn_out_low_q8_mpp( id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipeline, const ds4_gpu_mul_mm_id_args *mm_args, id<MTLBuffer> src0, NSUInteger src0_off, id<MTLBuffer> src1, NSUInteger src1_off, id<MTLBuffer> dst, NSUInteger dst_off);
int ds4_gpu_encode_swiglu_flat( id<MTLCommandBuffer> cb, id<MTLBuffer> gate, NSUInteger gate_off, id<MTLBuffer> up, NSUInteger up_off, id<MTLBuffer> out, NSUInteger out_off, uint32_t n);
int ds4_gpu_encode_moe_swiglu_weight( id<MTLCommandBuffer> cb, id<MTLBuffer> gate, NSUInteger gate_off, id<MTLBuffer> up, NSUInteger up_off, id<MTLBuffer> mid, NSUInteger mid_off, id<MTLBuffer> weights, NSUInteger weights_off, uint32_t width, uint32_t rows, float clamp_value, bool mid_f16);
int ds4_gpu_encode_moe_sum_experts( id<MTLCommandBuffer> cb, id<MTLBuffer> experts, NSUInteger experts_off, id<MTLBuffer> out, NSUInteger out_off, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens);
int ds4_gpu_encode_get_rows_i32_token_rows( id<MTLCommandBuffer> cb, id<MTLBuffer> table, NSUInteger table_off, id<MTLBuffer> tokens, NSUInteger tokens_off, const int32_t *token_inline, id<MTLBuffer> selected, NSUInteger selected_off, uint32_t hash_rows, uint32_t n_cols, uint32_t n_tokens);
int ds4_gpu_encode_get_rows_f32_router_weights( id<MTLCommandBuffer> cb, id<MTLBuffer> probs, NSUInteger probs_off, id<MTLBuffer> selected, NSUInteger selected_off, id<MTLBuffer> weights, NSUInteger weights_off, uint32_t n_expert, uint32_t n_expert_used, uint32_t n_tokens);
int ds4_gpu_encode_sum_rows_f32( id<MTLCommandBuffer> cb, id<MTLBuffer> src, NSUInteger src_off, id<MTLBuffer> dst, NSUInteger dst_off, uint32_t width, uint32_t rows);
int ds4_gpu_encode_router_select( id<MTLCommandBuffer> cb, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, id<MTLBuffer> logitsbuf, NSUInteger logits_off, id<MTLBuffer> biasbuf, NSUInteger bias_off, id<MTLBuffer> hashbuf, NSUInteger hash_off, id<MTLBuffer> tokensbuf, NSUInteger tokens_off, const int32_t *single_token, uint32_t hash_rows, uint32_t n_tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, bool has_bias, bool hash_mode);
id<MTLBuffer> ds4_gpu_route_clamp_ensure_buf(void);
int ds4_gpu_router_select_tensor( ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, uint32_t layer);
int ds4_gpu_routed_moe_batch_tensor( ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts, const ds4_gpu_residual_set *residual, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, uint32_t slot_start, /* TP Phase-3 batch split: owned slot range */ uint32_t slot_count, /* 0 or n_expert => no split (full) */ bool *mid_is_f16);

/* 热路径微 helper: static inline 原样搬移(勿变跨 TU 导出)。 */
static inline DS4MetalTensor *ds4_gpu_tensor_obj(ds4_gpu_tensor *tensor) {
    return (__bridge DS4MetalTensor *)tensor;
}

static inline const DS4MetalTensor *ds4_gpu_tensor_const_obj(const ds4_gpu_tensor *tensor) {
    return (__bridge const DS4MetalTensor *)tensor;
}

static inline id<MTLBuffer> ds4_gpu_tensor_buffer(const ds4_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static inline NSUInteger ds4_gpu_tensor_offset(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static inline id<MTLCommandBuffer> ds4_gpu_command_buffer(int *owned) {
    if (g_batch_cb) {
        *owned = 0;
        return g_batch_cb;
    }
    *owned = 1;
    return [g_queue commandBuffer];
}

static inline id<MTLComputeCommandEncoder> ds4_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        if (!g_batch_enc) g_batch_enc = [cb computeCommandEncoder];
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static inline void ds4_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static inline uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static inline double ds4_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static inline double ds4_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static inline NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static inline float ds4_gpu_negative_infinity(void) {
    union { uint32_t u; float f; } v = { 0xff800000u };
    return v.f;
}

static inline float ds4_gpu_positive_infinity(void) {
    union { uint32_t u; float f; } v = { 0x7f800000u };
    return v.f;
}

static inline NSUInteger ds4_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static inline NSUInteger ds4_gpu_bin_threads(uint32_t width, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)width && nth < nth_max) nth *= 2u;
    return nth ? nth : 1u;
}

#include "metal_expert.h"

#endif /* DS4_METAL_INTERNAL_H */
