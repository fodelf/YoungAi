/* =========================================================================
 * ds4.c - DeepSeek V4 inference engine.
 * =========================================================================
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * DeepSeek V4 tensor layouts, CPU reference kernels, the whole-model Metal
 * graph driver, and tokenizer wiring.  Model shape selection is intentionally
 * narrow: validation accepts the known Flash and Pro layouts and fails early
 * for anything else.
 *
 * Loading is mmap based.  The loader parses only the GGUF header, metadata
 * table, and tensor directory.  Tensor data stays in the kernel page cache
 * until inference touches it, or until Metal wraps slices of the mapping as
 * no-copy MTLBuffers.
 */
/* core_internal.h — ds4.c 机械拆分(重构阶段4)后 src/core 各切片的共享内部头。
 * 叠加在 ds4_internal.h 之上; 公共头(ds4.h/ds4_internal.h/ds4_gpu.h)不改。
 * 上面的文件头注释原样保留自拆分前的 ds4.c。 */
#ifndef DS4_CORE_INTERNAL_H
#define DS4_CORE_INTERNAL_H

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_multimodal.h"
#include "ds4_zchain.h"
#include "ds4_spatial.h"
#include "ds4_css.h"

#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define DS4_NEG_INF (-1.0e30f)
#define DS4_POS_INF ( 1.0e30f)
#define DS4_DEFAULT_RMS_EPS ( 1.0e-6f)
#define DS4_DEFAULT_HC_EPS  ( 1.0e-6f)
#define DS4_DEFAULT_SWIGLU_CLAMP_EXP    (10.0f)
#define DS4_DEFAULT_ROPE_FREQ_BASE      (10000.0f)
#define DS4_DEFAULT_ROPE_SCALE_FACTOR   (16.0f)
#define DS4_DEFAULT_ROPE_YARN_BETA_FAST (32.0f)
#define DS4_DEFAULT_ROPE_YARN_BETA_SLOW (1.0f)
#define DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define DS4_DEFAULT_ROPE_ORIG_CTX       UINT64_C(65536)


/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */
#define DS4_THINK_MAX_MIN_CONTEXT 393216u

#if defined(__GNUC__) || defined(__clang__)
#define DS4_MAYBE_UNUSED __attribute__((unused))
#else
#define DS4_MAYBE_UNUSED
#endif

#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */



#define DS4_GIB (1024.0 * 1024.0 * 1024.0)


#define DS4_DSPARK_WIN   128u
#define DS4_DSPARK_BLK   5u
#define DS4_DSPARK_NOISE 128799

/* stages 位掩码(2026-08-21 请求批处理): 1=投影段(hc mix/norm/q/kv, 行无关)
 * 2=KV 段(压缩器/索引器/注意力, 每行要自己的缓存) 4=出口段(inv_rope/o 投影/hc post,
 * 行无关)。默认 7 = 整半层, 与改造前逐字节一致。多会话批解码用 1|4 在共享图上批算
 * 行无关部分, 用 2 逐会话在各自图上算 KV 段。 */
#define DS4_ATTN_STAGE_PRE   1u
#define DS4_ATTN_STAGE_KV    2u
#define DS4_ATTN_STAGE_POST  4u
#define DS4_ATTN_STAGE_ALL   7u
/* 8 = 投影段里跳过 q/kv 的 rope 与 kv 的 fp8 量化(它们依赖"这一行的真实位置")。
 * 多会话批解码时 N 行来自不同会话、位置各不相同, 而 rope 入口只吃一个 pos0 —— 于是
 * 投影段先不做 rope, 由驱动按行用真实位置逐行补(rope 是逐元素的, 逐行调同一个入口,
 * 数值与整批调用逐位一致)。 */
#define DS4_ATTN_STAGE_NOROPE 8u

#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)


#include "ds4_internal.h"
#include "core_types.h"
#include "core_inline.h"
#include "core_gpu_graph.h"

struct ds4_session {
    ds4_engine *engine;
    ds4_dist_session *distributed;
#ifndef DS4_NO_GPU
    ds4_gpu_graph graph;
#endif
    ds4_kv_cache cpu_cache;
    ds4_cpu_decode_scratch cpu_scratch;
    token_vec checkpoint;
    float *logits;
    float *mtp_logits;
    int mtp_draft_token;
    uint64_t mtp_probe_total;
    uint64_t mtp_probe_hit;
    ds4_session_progress_fn progress;
    void *progress_ud;
    ds4_session_progress_fn display_progress;
    void *display_progress_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    /* Anticycle/request-penalty generation boundary: first position of the
     * CURRENT response's generated region. -1 = unmarked, and the penalty
     * core then falls back to 0 = the WHOLE context is scanned (legacy
     * behavior; also the deliberate dual-host-parity default, see
     * repeat_penalize_buf). Frontends that mark it after prefill
     * (ds4_session_mark_generation_start / _set_generation_start) scope the
     * anticycle bans and request penalties to [gen_start, end) only, so
     * verbatim prompt quoting stops being banned. Reset to -1 whenever the
     * checkpoint is rebuilt. The env freq-penalty window intentionally
     * IGNORES this boundary (2026-07-06 mono verdict; see
     * repeat_penalize_core). */
    int repeat_gen_start;
    /* Sampling-lane policy (ds4.h DS4_LANE_*), frontend-declared request
     * shape. Non-FREE lanes bypass ALL penalties (env freq, anticycle bans,
     * request penalties): forced tool-call syntax and copy-constrained
     * emission must see raw logits. Reset to FREE at creation/invalidate --
     * sessions are reused across requests and a stale lane must not leak. */
    int lane;
    /* Per-request OpenAI-style penalties (frequency/presence), FREE lane
     * only, counted over the generated region [repeat_gen_start, end). May be
     * negative (OpenAI allows [-2,2] to encourage repetition). (0,0) = off;
     * frontends re-declare on every request. */
    float req_freq;
    float req_presence;
    /* Greedy speculative acceptance (copy-spec / MTP argmax gating) is only
     * distribution-preserving at temperature 0; frontends clear this for
     * sampled requests. Default 1 = legacy greedy-ok. */
    int spec_greedy;
    bool mtp_draft_valid;
    /* Adaptive copy-spec draft length (single-machine prompt-lookup speculation).
     * Converges toward the length the target reliably accepts: grow on a full
     * accept, shrink to the observed accept on a partial. 0 = not yet primed.
     * This is what keeps copy-spec a net win on predictable text and a no-op on
     * unpredictable text, with no tuning knob. */
    int cs_draft_len;
    /* Cooldown: after a low-payoff speculation (the per-call frontier snapshot
     * cost only pays off when many tokens are accepted), skip speculating for a
     * few steps so unpredictable text never pays the snapshot tax. One probe
     * leaks through each cooldown so it re-arms the moment text becomes
     * predictable again. */
    int cs_cooldown;
    /* Go-trie drafter cooldown, deliberately separate from cs_cooldown: a trie
     * partial accept must not suppress the (independently profitable) n-gram
     * drafter, and vice versa. Measured on go1b/A3: sharing the cooldown +
     * letting trie partials shrink cs_draft_len cost -25% gen t/s by starving
     * n-gram batches (fable5.md prompt2 table). */
    int gt_cooldown;
    /* Reference-corpus drafter cooldown -- its own for the same
     * non-contamination reason as gt_cooldown. */
    int rc_cooldown;
};

/* ---- 跨文件全局与函数声明(拆分工序新增; 定义处已去 static) ---- */
extern const char DS4_REASONING_EFFORT_MAX_PREFIX[];
extern uint32_t g_ds4_compress_ratios[DS4_MAX_LAYER];
extern uint32_t g_requested_threads;
extern const gguf_type_info gguf_types[43];
extern bool g_model_open_arm_env_defaults;
extern int g_prefill_chunk_cuda;
extern bool g_vq_experts_blob;
extern pthread_once_t iq2xxs_signed_grid_once;
void iq2xxs_signed_grid_init(void);
extern int g_dspark_ready_global;
extern int g_ds4_spec_enabled;
extern const char *g_ds4_draft_gguf_path;
extern const ds4_dspark_weights *g_dspark_bound_for_prefill;
extern ds4_model *g_draft_model;
void matvec_any(float *out, const ds4_model *m, const ds4_tensor *w, const float *x);
void output_logits_one( float * logits, const ds4_model * model, const ds4_weights * weights, const float * inp_hc);
void output_logits_one_decode_scratch( float * logits, const ds4_model * model, const ds4_weights * weights, const float * inp_hc, ds4_cpu_decode_scratch * scratch);
int sample_argmax(const float *logits, uint32_t n_vocab);
bool cpu_directional_steering_enabled( const float *dirs, float scale);
void cpu_directional_steering_project_rows( float *x, const float *dirs, uint32_t il, uint32_t rows, float scale);
void residual_free(struct ds4_residual *r);
void repeat_penalize_buf(ds4_session *s, float *logits, uint32_t end);
int session_penalties_active(const ds4_session *s);
void print_vec_stats(const char *name, const float *x, uint64_t n);
bool ds4_backend_uses_graph(ds4_backend backend);
uint32_t ds4_layer_compress_ratio(uint32_t il);
uint32_t ds4_expected_layer_compress_ratio(uint32_t il);
char *byte_encode(ds4_str in, uint64_t *out_len);
bool compressor_decode_one( float * out_comp, const ds4_model * model, const ds4_tensor * wkv, const ds4_tensor * wgate, const ds4_tensor * ape, const ds4_tensor * norm, const float * x, float * state_kv, float * state_score, uint32_t head_dim, uint32_t compress_ratio, uint32_t il, uint32_t pos);
bool compressor_decode_one_decode_scratch( float * out_comp, const ds4_model * model, const ds4_tensor * wkv, const ds4_tensor * wgate, const ds4_tensor * ape, const ds4_tensor * norm, const float * x, float * state_kv, float * state_score, uint32_t head_dim, uint32_t compress_ratio, uint32_t il, uint32_t pos, ds4_cpu_decode_scratch * scratch);
void compressor_pool_decode_state( float * out, float * state_kv, float * state_score, uint32_t head_dim, uint32_t compress_ratio);
void config_validate_model(const ds4_model *m);
void cpu_decode_scratch_free(ds4_cpu_decode_scratch *scratch);
void cpu_decode_scratch_init(ds4_cpu_decode_scratch *scratch, uint32_t ctx_size);
ds4_cursor cursor_at(const ds4_model *m, uint64_t pos);
bool cursor_read(ds4_cursor *c, void *dst, uint64_t n);
bool cursor_string(ds4_cursor *c, ds4_str *s);
bool cursor_u32(ds4_cursor *c, uint32_t *v);
bool cursor_u64(ds4_cursor *c, uint64_t *v);
void ds4_acquire_instance_lock(void);
void ds4_alloc_guard_begin(const char *phase);
void ds4_alloc_guard_end(void);
uint32_t ds4_default_prefill_cap_for_prompt(int prompt_len);
uint32_t ds4_default_raw_cap(uint32_t ctx_size);
void ds4_die_errno(const char *what, const char *path);
void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx);
void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows);
void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);
void ds4_release_instance_lock(void);
int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp, char *err, size_t errlen);
bool ds4_session_is_cpu(const ds4_session *s);
int ds4_session_slice_check_timeline( ds4_session *s, const int *tokens, uint32_t n_tokens, uint32_t pos0, char *err, size_t errlen);
DS4_MAYBE_UNUSED void ds4_session_slice_commit_timeline(ds4_session *s, const int *tokens, uint32_t n_tokens);
int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
char *ds4_strdup(const char *s);
void ds4_threads_shutdown(void);
void ds4_vec_dot_iq2_xxs_pair_q8_K( int n, float *s0, float *s1, const block_iq2_xxs *x0, const block_iq2_xxs *x1, const block_q8_K *y);
void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y);
void dspark_bind_with_draft(ds4_dspark_weights *w, const ds4_model *m, bool graph_backend);
void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot);
void dsv4_indexer_qat_row_inplace_cpu(float *x, uint32_t head_dim);
void dsv4_indexer_qat_rows_inplace_cpu(float *x, uint32_t rows, uint32_t head_dim);
void embed_prompt( const ds4_model * model, const ds4_weights * weights, const token_vec * tokens, uint32_t n_embd, float * out);
void embed_token_f16(const ds4_model *m, const ds4_weights *w, int token, float *out);
void f16_round_inplace_cpu(float *x, uint32_t n);
void forward_token_raw_swa_cpu_decode_scratch( float * logits, const ds4_model * model, const ds4_weights * weights, ds4_kv_cache * cache, int token, uint32_t pos, const float * steering_dirs, float steering_attn_scale, float steering_ffn_scale, ds4_cpu_decode_scratch * scratch);
int generate_raw_swa_cpu( const ds4_model * model, const ds4_vocab * vocab, const ds4_weights * weights, const token_vec * prompt, int n_predict, int ctx_size, const float * directional_steering_dirs, float directional_steering_attn, float directional_steering_ffn, ds4_token_emit_fn emit, ds4_generation_done_fn done, void * emit_ud, ds4_session_progress_fn progress, void * progress_ud);
uint64_t hash_bytes(const void *ptr, uint64_t len);
void hc_from_plain_embedding(float *out_hc, const float *x, uint32_t n_embd, uint32_t n_hc);
void hc_post_batch( float * out_hc, const float * block_out, const float * residual_hc, const float * post, const float * comb, uint32_t n_tok, uint32_t n_embd, uint32_t n_hc);
void hc_post_one( float * out_hc, const float * block_out, const float * residual_hc, const float * post, const float * comb, uint32_t n_embd, uint32_t n_hc);
void hc_post_sum_batch( float * out_hc, const float * moe, const float * shared, const float * residual_hc, const float * post, const float * comb, uint32_t n_tok, uint32_t n_embd, uint32_t n_hc);
void hc_pre_from_state_one( const ds4_model * model, const ds4_tensor * fn, const ds4_tensor * scale_tensor, const ds4_tensor * base_tensor, const float * residual_hc, float * out, float * post, float * comb);
void hc_pre_from_state_one_scratch( const ds4_model * model, const ds4_tensor * fn, const ds4_tensor * scale_tensor, const ds4_tensor * base_tensor, const float * residual_hc, float * out, float * post, float * comb, float * flat, bool serial_fn);
void hc_pre_norm_batch( const ds4_model * model, const ds4_tensor * fn, const ds4_tensor * scale, const ds4_tensor * base, const ds4_tensor * norm_w, const float * inp_hc, float * residual_hc, float * cur, float * norm, float * post, float * comb, uint32_t n_tok);
void hc_weighted_sum_one( float * out, const float * x, const float * weights, uint32_t n_embd, uint32_t n_hc);
void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps);
bool *indexer_allowed_decode_one( const ds4_model * model, const ds4_layer_weights * layer, const float * cur, const float * qr_norm, const float * index_comp, uint32_t n_comp, uint32_t il, uint32_t pos);
bool *indexer_allowed_decode_one_decode_scratch( const ds4_model * model, const ds4_layer_weights * layer, const float * cur, const float * qr_norm, const float * index_comp, uint32_t n_comp, uint32_t il, uint32_t pos, ds4_cpu_decode_scratch * scratch);
void kv_cache_free(ds4_kv_cache *cache);
void kv_cache_init(ds4_kv_cache *cache, uint32_t ctx_size, uint32_t raw_cap);
void kv_cache_push_comp(float *rows, uint32_t *n_rows, uint32_t cap_rows, uint32_t row_dim, const float *kv);
void kv_cache_push_raw(ds4_layer_cache *cache, const float *kv);
void layer_attention_mixed_one( float * out_heads, const ds4_model * model, const ds4_layer_weights * layer, const float * q, const float * raw_kv, uint32_t n_raw, const float * comp_kv, uint32_t n_comp, const bool * comp_allowed);
void layer_attention_mixed_one_decode_scratch( float * out_heads, const ds4_model * model, const ds4_layer_weights * layer, const float * q, const float * raw_kv, uint32_t n_raw, const float * comp_kv, uint32_t n_comp, const bool * comp_allowed, ds4_cpu_decode_scratch * scratch);
void layer_attention_one( float * out_heads, const ds4_model * model, const ds4_layer_weights * layer, const float * q, const float * kv);
void layer_attention_rows_one( float * out_heads, const ds4_model * model, const ds4_layer_weights * layer, const float * q, const float * kv_rows, uint32_t n_kv);
void layer_attn_norm_one( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * x);
DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio);
void layer_ffn_one( float * out_hc, const ds4_model * model, const ds4_layer_weights * layer, const float * inp_hc, uint32_t il, int token, const float * steering_dirs, float steering_scale, bool trace);
void layer_ffn_one_decode_scratch( float * out_hc, const ds4_model * model, const ds4_layer_weights * layer, const float * inp_hc, uint32_t il, int token, const float * steering_dirs, float steering_scale, ds4_cpu_decode_scratch * scratch);
void layer_grouped_out_one( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * heads);
void layer_grouped_out_one_decode_scratch( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * heads, ds4_cpu_decode_scratch * scratch);
void layer_hash_router_weights_one( float weights_out[DS4_MAX_EXPERT_USED], const ds4_model * model, const ds4_layer_weights * layer, const float * x, const int selected[DS4_MAX_EXPERT_USED]);
void layer_hash_selected_experts( int selected[DS4_MAX_EXPERT_USED], const ds4_model *model, const ds4_layer_weights *layer, int token);
DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio);
void layer_kv_projection_normed_one( const ds4_model * model, const ds4_layer_weights * layer, const float * normed, float * kv);
void layer_kv_projection_normed_one_decode_scratch( const ds4_model * model, const ds4_layer_weights * layer, const float * normed, float * kv, ds4_cpu_decode_scratch * scratch);
void layer_q_projection_normed_one( const ds4_model * model, const ds4_layer_weights * layer, const float * norm, float * q);
void layer_q_projection_with_lora_one( const ds4_model * model, const ds4_layer_weights * layer, const float * norm, float * q, float * qr_norm);
void layer_q_projection_with_lora_one_decode_scratch( const ds4_model * model, const ds4_layer_weights * layer, const float * norm, float * q, float * qr_norm, ds4_cpu_decode_scratch * scratch);
float layer_rope_freq_base(uint32_t il);
float layer_rope_freq_scale(uint32_t il);
void layer_routed_moe_batch( float * moe, const ds4_model * model, const ds4_layer_weights * layer, const float * norm, const int * token_ids, uint32_t n_tok, uint32_t il, float clamp);
void layer_routed_moe_one( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * x, uint32_t il, int token, float clamp, bool trace);
void layer_routed_moe_one_prealloc( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * x, uint32_t il, int token, float clamp, float * mid_all, block_q8_K * xq, block_q8_K * midq);
void layer_shared_ffn_batch( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * x, uint32_t n_tok);
void layer_shared_ffn_one( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * x);
void layer_shared_ffn_one_decode_scratch( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * x, ds4_cpu_decode_scratch * scratch);
void layer_topk_selected_experts( int selected[DS4_MAX_EXPERT_USED], float expert_weight[DS4_MAX_EXPERT_USED], const ds4_model *model, const ds4_layer_weights *layer, const float *x, const float *logit_bias);
void matmul_q8_0_batch( float * out, const ds4_model * m, const ds4_tensor * w, const float * x, uint64_t n_tok);
void matmul_q8_0_grouped_batch( float * out, const ds4_model * m, const ds4_tensor * w, const float * x, uint64_t n_tok, uint32_t n_groups, uint64_t group_dim, uint64_t rank);
void matmul_q8_0_pair_batch( float * out0, float * out1, const ds4_model * m, const ds4_tensor * w0, const ds4_tensor * w1, const float * x, uint64_t n_tok);
void matvec_any_decode_scratch( float * out, const ds4_model * m, const ds4_tensor * w, const float * x, ds4_cpu_decode_scratch * scratch);
void matvec_f16(float *out, const ds4_model *m, const ds4_tensor *w, const float *x);
void matvec_f16_serial(float *out, const ds4_model *m, const ds4_tensor *w, const float *x);
void matvec_iq2_xxs_expert_pair_prequant( float *out0, float *out1, const ds4_model *m, const ds4_tensor *w0, const ds4_tensor *w1, const block_q8_K *xq, uint32_t expert);
void matvec_iq2_xxs_experts_mid_prequant( float *mid, const ds4_model *m, const ds4_tensor *gate_w, const ds4_tensor *up_w, const block_q8_K *xq, const int *selected, const float *expert_weight, int n_expert, float clamp);
void matvec_q2_k_expert( float *out, const ds4_model *m, const ds4_tensor *w, const float *x, uint32_t expert);
void matvec_q2_k_experts_accum_prequant( float *out, const ds4_model *m, const ds4_tensor *w, const block_q8_K *xq, const int *selected, int n_expert);
void matvec_q8_0(float *out, const ds4_model *m, const ds4_tensor *w, const float *x);
void matvec_q8_0_decode_scratch( float * out, const ds4_model * m, const ds4_tensor * w, const float * x, ds4_cpu_decode_scratch * scratch);
void matvec_q8_0_grouped_rows( float * out, const ds4_model * m, const ds4_tensor * w, const float * x, uint32_t n_groups, uint64_t group_dim, uint64_t rank);
void matvec_q8_0_grouped_rows_decode_scratch( float * out, const ds4_model * m, const ds4_tensor * w, const float * x, uint32_t n_groups, uint64_t group_dim, uint64_t rank, ds4_cpu_decode_scratch * scratch);
void matvec_q8_0_pair_decode_scratch( float * out0, float * out1, const ds4_model * m, const ds4_tensor * w0, const ds4_tensor * w1, const float * x, ds4_cpu_decode_scratch * scratch);
void matvec_q8_0_pair_prequant( float * out0, float * out1, const ds4_model * m, const ds4_tensor * w0, const ds4_tensor * w1, const int8_t * xq, const float * xscale);
uint32_t model_expert_kept_count(const ds4_model *m, uint32_t il);
bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out);
bool model_get_f32_compat(const ds4_model *m, const char *key, float *out);
bool model_get_string(const ds4_model *m, const char *key, ds4_str *out);
bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out);
bool model_get_u64(const ds4_model *m, const char *key, uint64_t *out);
bool model_get_u64_compat(const ds4_model *m, const char *key, uint64_t *out);
void model_summary(const ds4_model *m);
void model_warm_weights(const ds4_model *m);
int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen);
void payload_set_err(char *err, size_t errlen, const char *msg);
DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen);
void prefill_layer_major_cpu( float * logits, const ds4_model * model, const ds4_weights * weights, ds4_kv_cache * cache, const token_vec * prompt, const float * steering_dirs, float steering_attn_scale, float steering_ffn_scale);
void print_top_logits( FILE * fp, const char * label, const ds4_vocab * vocab, const float * logits, uint32_t n_vocab, int k);
void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n);
void quantize_q8_0_activation_batch( const float *x, int8_t *xq, float *xscale, uint64_t n_tok, uint64_t in_dim);
bool read_f32_binary_file(const char *path, float *data, uint64_t n);
bool required_bool(const ds4_model *m, const char *key);
float required_f32(const ds4_model *m, const char *key);
ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer);
uint32_t required_u32(const ds4_model *m, const char *key);
struct ds4_residual *residual_load(const char *path, bool metal_mapping);
void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps);
void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps);
void rope_tail_layer_inplace( float * x, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t il, bool inverse);
uint64_t routed_expert_gate_off(const ds4_layer_weights *l);
uint64_t routed_expert_in_dim(const ds4_layer_weights *l);
uint64_t routed_expert_mid_dim(const ds4_layer_weights *l);
uint32_t routed_expert_quant_type(const ds4_layer_weights *l);
DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const ds4_tensor *t);
uint64_t routed_expert_up_off(const ds4_layer_weights *l);
int sample_top_p_min_p( const float *logits, uint32_t n_vocab, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
uint32_t session_cpu_comp_cap(const ds4_session *s);
uint64_t session_cpu_payload_live_tensor_bytes(const ds4_session *s);
uint32_t session_cpu_raw_live_rows(const ds4_session *s);
void session_cpu_reset_cache(ds4_session *s);
void session_reset_request_policy(ds4_session *s);
bool skip_value(ds4_cursor *c, uint32_t type, int depth);
void sleep_sec(double sec);
bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value);
void table_init(str_i32_table *t, uint64_t expected);
void table_put(str_i32_table *t, ds4_str key, int value);
float tensor_1d_value(const ds4_model *m, const ds4_tensor *t, uint64_t i);
float tensor_2d_value(const ds4_model *m, const ds4_tensor *t, uint64_t x, uint64_t y);
ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer);
bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
const gguf_type_info *tensor_type(uint32_t type);
const char *tensor_type_name(uint32_t type);
void token_vec_free(token_vec *tv);
void token_vec_push(token_vec *tv, int token);
int utf8_len_from_first_byte(uint8_t c);
void vocab_free(ds4_vocab *vocab);
void vocab_load(ds4_vocab *vocab, const ds4_model *model);
struct ds4_residual *vq_dir_load(const char *dir);
struct ds4_residual *vq_model_load(const ds4_model *m);
void weights_bind(ds4_weights *w, const ds4_model *m);
void weights_free(ds4_weights *w);
bool write_f32_binary_file(const char *path, const float *data, uint64_t n);
void *xmalloc_zeroed(size_t n, size_t size);
void *xrealloc(void *ptr, size_t size);
struct ds4_zchain *zchain_from_model(const ds4_model *m);
bool accelerator_cache_model_tensors(ds4_backend backend, const ds4_model *m);
bool cpu_load_directional_steering(ds4_engine *e);
DS4_MAYBE_UNUSED void ds4_l1_budget_gate(uint64_t resident_model_bytes, uint64_t kv_and_scratch_bytes);
void dump_tokens(const ds4_vocab *vocab, const token_vec *tokens);
void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const token_vec *tokens);
void forward_first_token_cpu( float * out_hc, const ds4_model * model, const ds4_weights * weights, int token);
void kv_cache_finish_prefill_states(ds4_kv_cache *cache, uint32_t n_tokens);
void layer_attention_raw_swa_batch( float * after_attn_hc, const ds4_model * model, const ds4_layer_weights * layer, ds4_layer_cache * cache, const float * inp_hc, uint32_t n_tok, uint32_t il, uint32_t pos0, const float * steering_dirs, float steering_scale);
void layer_attn_pre_one( const ds4_model * model, const ds4_layer_weights * layer, const float * token_embd, float * out, float * residual_hc, float * post, float * comb);
void layer_ffn_shared_batch( float * out_hc, const ds4_model * model, const ds4_layer_weights * layer, const float * inp_hc, const int * token_ids, uint32_t n_tok, uint32_t il, const float * steering_dirs, float steering_scale);
void layer_forward_self_one( float * out_hc, const ds4_model * model, const ds4_layer_weights * layer, const float * inp_hc, uint32_t il, uint32_t pos, int token);
void layer_grouped_out_batch( float * out, const ds4_model * model, const ds4_layer_weights * layer, const float * heads, uint32_t n_tok);
int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen);
ds4_tensor *routed_down_shadow(uint32_t il);
void table_free(str_i32_table *t);
const uint8_t *tensor_expert_bytes( const ds4_model *m, const ds4_tensor *w, uint32_t expert, uint64_t *in_dim, uint64_t *out_dim, uint64_t *row_bytes);
void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text, token_vec *out);
DS4_MAYBE_UNUSED bool weights_model_map_spans( const ds4_weights *w, uint32_t layer_start, uint32_t layer_end, bool include_output, bool include_token_embd, ds4_model_map_span_vec *spans);
bool weights_model_map_spans_split( const ds4_weights *w, ds4_model_map_span_vec *backbone, ds4_model_map_span_vec *experts);
bool weights_model_map_spans_split_slice( const ds4_weights *w, uint32_t layer_start, uint32_t layer_end, bool include_output, bool include_token_embd, ds4_model_map_span_vec *backbone, ds4_model_map_span_vec *experts);
void weights_validate_layout(const ds4_model *m, const ds4_weights *w);
void ds4_eval_ids_run(ds4_engine *e);
void ds4_multi_bench_run(ds4_engine *e);
void matvec_q8_0_worker(void *vctx, uint64_t r0, uint64_t r1);
void matvec_q8_0_pair_worker(void *vctx, uint64_t r0, uint64_t r1);
void matvec_q8_0_grouped_worker(void *vctx, uint64_t r0, uint64_t r1);
void matmul_q8_0_grouped_batch_worker(void *vctx, uint64_t r0, uint64_t r1);
void matmul_q8_0_batch_worker(void *vctx, uint64_t r0, uint64_t r1);
void matmul_q8_0_pair_batch_worker(void *vctx, uint64_t r0, uint64_t r1);
void matvec_iq2_xxs_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1);
void quantize_mid_pairs_worker(void *vctx, uint64_t p0, uint64_t p1);
void matvec_q2_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1);
char *engine_mm_spatial_enrich(void *ctx, const char *modality, const char *text);
char *engine_mm_css_enrich(void *ctx, const char *modality, const char *text);
void engine_tokenize_cb(void *ctx, const char *text, int **toks, int *n);

#endif /* DS4_CORE_INTERNAL_H */
