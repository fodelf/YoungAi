#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);

int ds4_gpu_token_graph_begin(void);
void ds4_gpu_token_graph_set_pos(uint32_t pos);
int ds4_gpu_token_graph_end_launch(void);
/* decode 流水线: 命中预编码图直接发射 / GPU 忙时为 pos 预捕获下一图(CUDA-only) */
int ds4_gpu_token_graph_try_pending(int token, uint32_t pos, int need_logits);
/* decode 双流并发: mark(主流,MoE 前) → begin(shared 段切侧流) → join(汇合) */
int ds4_gpu_side_mark(void);
int ds4_gpu_side_begin(void);
int ds4_gpu_side_main(void);
int ds4_gpu_dspark_hc_mean_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *hc,
                                  uint32_t n_embd, uint32_t n_hc, uint32_t slot, uint32_t n_tokens);
int ds4_gpu_dspark_attn_tensor(ds4_gpu_tensor *heads,
                               const void *model_map, uint64_t model_size, uint64_t sinks_offset,
                               const ds4_gpu_tensor *q, const ds4_gpu_tensor *win_kv,
                               const ds4_gpu_tensor *blk_kv,
                               uint32_t n_win, uint32_t blk, uint32_t n_head, uint32_t head_dim,
                               uint32_t win_base, uint32_t win_cap);
int ds4_gpu_dspark_win_scatter_tensor(ds4_gpu_tensor *win, const ds4_gpu_tensor *rows,
                                      uint32_t n, uint32_t pos0, uint32_t win_rows, uint32_t dim);
int ds4_gpu_dspark_argmax_only_tensor(ds4_gpu_tensor *out_id,
                                      const ds4_gpu_tensor *logits_row, uint32_t vocab);
/* DSpark 置信头: c_k = sigmoid(w·[x_k ; W1[prev_k]]) —— 调度器按前缀存活率 ∏c 选验证长度 */
int ds4_gpu_dspark_confidence_tensor(ds4_gpu_tensor *out_conf, const ds4_gpu_tensor *x,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t conf_w_offset, uint64_t markov_w1_offset,
                                     const ds4_gpu_tensor *prev_ids,
                                     uint32_t dim, uint32_t rank, uint32_t vocab, uint32_t n_pos);

int ds4_gpu_dspark_markov_step_tensor(ds4_gpu_tensor *out_id, ds4_gpu_tensor *logits_row,
                                      const void *model_map, uint64_t model_size,
                                      uint64_t w1_offset, uint64_t w2_offset,
                                      const ds4_gpu_tensor *prev_id, uint32_t vocab, uint32_t rank);
int ds4_gpu_side_join(void);
int ds4_gpu_matmul_q4_K_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t off0, uint64_t off1,
                                    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                    const ds4_gpu_tensor *x);
int ds4_gpu_matmul_q2_K_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                    const void *model_map, uint64_t model_size,
                                    uint64_t off0, uint64_t off1,
                                    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                    const ds4_gpu_tensor *x);
int ds4_gpu_token_graph_precapture_begin(void);
int ds4_gpu_token_graph_precapture_end(uint32_t pos, int need_logits, int encode_ok);
int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

/* Tensor-parallel rendezvous. ds4_gpu_tp_signal_after_batch encodes a shared-
 * event signal at the tail of the current batch and returns the value to wait on
 * (0 on error); the caller flushes the batch so the GPU runs and fires it.
 * ds4_gpu_tp_host_wait blocks on the fast MTLSharedEvent path until that value,
 * making the just-encoded results host-visible without a full pipeline drain. */
uint64_t ds4_gpu_tp_signal_after_batch(void);
int ds4_gpu_tp_host_wait(uint64_t value);

/* GPU 跨度计时(verify 归因): span_begin/end 之间的 GPU 侧毫秒 */
void ds4_gpu_span_begin(void);
float ds4_gpu_span_end(void);
int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
/* 副模型(drafter gguf)整体 map 注册: 不动主模型状态, 只加一条 range */
int ds4_gpu_register_aux_model_map(const void *map, uint64_t size);
/* verify 批 CUDA 图: begin(slot=pos0&3) 开始捕获; end_launch(encode_ok) 收尾并发射。
 * 返回 1=已发射(图), 0=未启用(直发), -1=捕获失败(调用方需恢复状态后重编码直发)。 */
/* 数值护栏: 把张量里的非有限值(NaN/Inf)就地置 0 */
int ds4_gpu_sanitize_finite_tensor(ds4_gpu_tensor *t, uint64_t n_float);
/* 路由空槽消毒: selected 里的 -1/越界项归零并清其权重(避免 counts[-1] 越界与未写 mid 槽) */
int ds4_gpu_sanitize_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                                   uint32_t n_pairs, uint32_t n_total_expert);
int ds4_gpu_batch_graph_begin(int slot);
int ds4_gpu_batch_graph_end_launch(int encode_ok);
/* 批版 pair 融合(同输入两矩阵一发): 激活量化一次 + 行数合并解并行度 + 权重驻留。
 * 返回 0 = 不适用(调用方回退两次单发)。 */
int ds4_gpu_matmul_q2_K_pair_batch_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                          const void *model_map, uint64_t model_size,
                                          uint64_t off0, uint64_t off1,
                                          uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                          const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
/* When on!=0, the NEXT ds4_gpu_set_model_map_range wraps its views without
 * adding them to the GPU residency set (evictable mmap, not wired).  Auto-state;
 * set before the MTP draft map to keep it off the worker's wired budget. CPU
 * builds ignore it. */
void ds4_gpu_set_model_map_nonresident_hint(int on);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
/* Reduced-memory model loader. Identical to ds4_gpu_set_model_map_spans but each
 * span carries a resident flag: resident spans (backbone) are wired into the GPU
 * residency set; non-resident spans (routed experts) are still wrapped so the hot
 * path can resolve their buffers, but are kept out of the residency set so their
 * clean file-backed pages stay reclaimable under memory pressure. resident_flags
 * must be non-NULL with one entry per span. Used for DS4_METAL_EXPERT_OFFLOAD. */
int ds4_gpu_set_model_map_spans_split(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, const bool *resident_flags, uint32_t count, uint64_t max_tensor_bytes);
/* Dynamic routed-expert residency route. The host decides at load time whether
 * the fully-resident model fits the memory budget and pushes the verdict here so
 * the hot-path gather agrees (1 = stream/offload, 0 = keep resident). An explicit
 * DS4_METAL_EXPERT_OFFLOAD env always overrides this. CPU builds ignore it. */
void ds4_gpu_set_expert_offload(int enabled);
/* Device recommended max GPU working-set in bytes (0 if unknown / CPU build).
 * Used as the AUTO offload budget when DS4_MEM_BUDGET_MB is unset. */
uint64_t ds4_gpu_recommended_max_working_set_bytes(void);
/* Live GPU working-set (model wired + scratch). 0 on CPU build. */
uint64_t ds4_gpu_current_allocated_bytes(void);
/* P2.1 cross-layer router prediction prefetch (project.md): register one routed
 * MoE layer's router metadata so the backend can re-evaluate the next layer's
 * router on the CPU during decode and issue async read-ahead for the predicted
 * experts while the GPU is still computing the current layer.  gate_inp must be
 * the F16 [n_embd][n_expert] router matrix; probs_bias_offset is UINT64_MAX
 * when the model has no exp_probs_b bias.  Purely advisory: wrong predictions
 * waste read bandwidth but can never change inference results. */
/* hash_table_offset/k/rows describe the early layers' token-id hash routing
 * (ffn_gate_tid2eid I32 [k][n_vocab]): their expert set is an exact function
 * of the token id, so the backend can stage them with 100% accuracy as soon
 * as the token is known.  hash_table_offset == UINT64_MAX for score routing. */
int ds4_gpu_register_layer_router(const void *model_map, uint32_t layer, uint64_t gate_inp_offset, int gate_inp_is_f32, uint64_t probs_bias_offset, uint64_t gate_exps_offset, uint64_t up_exps_offset, uint64_t down_exps_offset, uint64_t gate_expert_bytes, uint64_t down_expert_bytes, uint32_t n_embd, uint32_t n_expert, uint64_t hash_table_offset, uint32_t hash_k, uint32_t hash_rows);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
/* CUDA decode: 启动期把 q8_0 权重 repack 成 scale/qs 分离平面(对齐 128bit 读)。
 * token graph capture 时 host dispatch 只跑一次, repack 表必须先建好。 */
int ds4_gpu_q8r_preload(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t in_dim, uint64_t out_dim);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

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
        float                   scale);

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
        float                   scale);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* TP row-parallel Q8_0 matvec (decode/n_tok=1). Computes a PARTIAL out[out_dim]
 * over the input-dim block range [0, in_dim_slice) of each weight row, rows
 * strided by in_dim_full. weight_offset is pre-shifted to the owned slice start;
 * x is the compacted [in_dim_slice]. All-reduce the result across TP peers to get
 * the full matvec (~1e-6 drift vs single-machine — inherent to row-parallel). */
int ds4_gpu_attention_output_q4k_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_q4k_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);

int ds4_gpu_matmul_q4_K_hc_expand_tensor(
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
        uint32_t                n_hc);

int ds4_gpu_matmul_q4_K_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* dense Q2_K matmul(2026-08-19 全q2 基座): 布局/语义与 q4_K 版逐参同。 */
int ds4_gpu_matmul_q2_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok);

/* q2_K 版批量 attn_output(2026-08-19 全q2): 参数与 q4k 版逐参同。 */
int ds4_gpu_attention_output_q2k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens);

/* 全q2 f16 影子注册(2026-08-19): q2_K 权重一次性 dequant→f16 device buffer,
 * 之后该 offset 的一切消费(range_ptr 命中)透明拿到 f16。rows/cols=行数/行长。 */
int ds4_gpu_register_q2k_f16_shadow(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t rows, uint64_t cols);

int ds4_gpu_matmul_q8_0_rowslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim_full,
        uint64_t                in_dim_slice,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x);

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
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

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
        uint64_t                n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

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
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);

int ds4_gpu_kv_rope_fp8_store_raw_tensor(
    ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache,
    uint32_t raw_cap, uint32_t raw_row, uint32_t head_dim, uint32_t n_rot,
    uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale,
    float ext_factor, float attn_factor, float beta_fast, float beta_slow);
int ds4_gpu_head_rms_norm_rope_tail_tensor(
    ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
    uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
    float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow, float eps);
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
        float             beta_slow);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

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
        float                   rms_eps);

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
        uint32_t                n_tokens);

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
        float                   rms_eps);

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
        float                   rms_eps);

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
        uint32_t                pos0);

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
        uint32_t                head_dim);

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
        uint32_t                head_dim);

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
        uint32_t                head_dim);

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
        uint32_t                head_dim);

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
        uint32_t                head_dim);

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
        uint32_t                head_dim);

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
        uint32_t                head_dim);

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
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

/* go1b "hidden variable z^L" four-loss correction (resident sidecar tensors).
 *
 * ds4_gpu_corr_router_bias adds the per-expert router-logit bias delta[e] to the
 * raw router logits (pre softplus/sqrt) BEFORE top-k selection, broadcast across
 * all n_tokens rows: logits[t][e] += delta[e]. Apply to score-routed layers only.
 *
 * ds4_gpu_corr_apply adds the per-selected-expert correction to the already-summed
 * routed-MoE output, in place:  out[t][d] += sum over selected e of
 *   ( U @ ( C[e] (.*) (V @ x[t]) ) )[d] + b[d] + beta[e].
 * x is the per-token FFN input fed to the experts (post-RMSNorm activation).
 * U is [d_model][d_l] row-major, V is [d_l][d_model] row-major, C is [n_expert][d_l]
 * row-major, b is [d_model], beta is [n_expert]; selected is [n_tokens][n_expert_used]
 * (original 0..n_expert-1 ids). One threadgroup per token; vx is recomputed per token. */
int ds4_gpu_corr_router_bias(
        ds4_gpu_tensor       *logits,
        const ds4_gpu_tensor *delta,
        uint32_t                n_expert,
        uint32_t                n_tokens);

int ds4_gpu_corr_apply(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t                d_model,
        uint32_t                d_l,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        uint32_t                n_tokens);

/* Store variant of corr_apply: writes the raw correction term into delta_out
 * (delta_out[t][d] = corr term) instead of accumulating into routed_out. The
 * decode shared-down HC fusion then adds routed[d]+delta[d] — the same fadd
 * the in-place kernel performed, so results are bit-identical, while the tiny
 * corr dispatch stops write-hazarding the hot routed_out buffer (a measured
 * ~23ms/layer full-pipeline bubble on Metal). Only used when
 * ds4_gpu_corr_delta_supported() returns nonzero AND the fused shared-down
 * consumer runs (single-host decode default); every other path keeps the
 * in-place kernel. */
int ds4_gpu_corr_apply_delta(
        ds4_gpu_tensor       *delta_out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t                d_model,
        uint32_t                d_l,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        uint32_t                n_tokens);

int ds4_gpu_corr_delta_supported(void);

/* go1b corr needs the ORIGINAL top-k expert ids, but the offload MoE remaps the
 * selected buffer to compact slots IN PLACE (ds4_gpu_remap_selected_to_slots)
 * before the corr reads it.  ds4_gpu_routed_moe_batch_tensor snapshots the
 * pre-remap ids into a persistent Shared buffer whenever it runs the go1b path;
 * the corr indexes C[e]/beta[e] from this copy instead of the corrupted buffer.
 * Returns NULL until the first go1b MoE call snapshots (callers fall back to the
 * live selected tensor). Layout matches selected: [n_tokens][n_expert_used]. */
ds4_gpu_tensor *ds4_gpu_corr_saved_selected(void);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

/* Reduced-expert (keep-map) routing support. A shrunken model's expert tensors
 * only carry the kept rows, but the router still selects in the full 256-wide
 * original id space. ds4_gpu_set_expert_keep_lut() uploads the per-layer
 * original-id -> compact-slot table (n_layer * 256 int16, -1 == dropped) into a
 * small resident GPU buffer once at load. ds4_gpu_translate_expert_ids() rewrites
 * a `selected` tensor from original ids to compact slots in place, between router
 * selection and the routed-MoE matvec. Both are no-ops for a full model (no LUT
 * set). Returns 1 on success, 0 on failure. */
int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer);
int ds4_gpu_translate_expert_ids(
        ds4_gpu_tensor       *selected,
        uint32_t                layer,
        uint32_t                n_expert_used,
        uint32_t                n_tokens,
        uint32_t                n_total_expert);

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
        const ds4_gpu_tensor *logits,
        uint32_t                layer);

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
        uint32_t                n_tokens,
        uint32_t                layer);

/* Optional 1-bit residual expert weights (go1b) for the routed MoE: a second 1-bit
 * layer Q1(W-Q1(W)) summed into each expert matmul. NULL/gate=NULL => no residual.
 * The buffers are RESIDENT GPU copies of the _res go1b tensors (a wrapped mmap view
 * of an offload model reads as ZEROS on the GPU, so the residual MUST be resident,
 * exactly like the corr sidecar). Strides/dims reuse the base expert args. */
typedef struct {
    /* CPU pointers to the go1b residual expert weights (sidecar mmap). The decode
     * path CPU-gathers the active experts into a compacted scratch (like the base
     * offload gather), then runs the go1b mm_id mapped-tile matmul over them. */
    const void *gate_ptr;   /* blk.L.ffn_gate_exps_res weights (K experts if sparse) */
    const void *up_ptr;     /* blk.L.ffn_up_exps_res weights */
    const void *down_ptr;   /* blk.L.ffn_down_exps_res weights */
    /* Sparse residual: lut[expert_id] = slot in [0,K) or -1 (no residual for that
     * expert). NULL => dense (gate_ptr indexed directly by expert id, K=256). */
    const float *lut;
    /* Nonzero when the sidecar tensors are go2b (type 41: offline-merged
     * base+residual, exact 4-level sum). The metal batch path then routes hot
     * picks through a single go2b matmul pass and masks them out of the base
     * pass, instead of the legacy base+residual add (3 extra matmuls/layer). */
    int merged2b;
    /* Nonzero when gate_ptr points at a v2.2 VQ layer blob (DQVL: 表+DQVQ 载荷).
     * The metal path dequants every active expert to an f16 scratch at gather
     * (cold w2 expanded from the base go1b bytes) and runs the F16W mm_id. */
    int vq;
    /* Total DQVL blob bytes when vq!=0. The CUDA path uses it to relocate the
     * host-mmap blob pointer onto the HBM arena copy (the startup span cache
     * already holds these bytes; reading the mmap original would re-fault pages
     * that were madvise(DONTNEED)d after the copy). */
    uint64_t vq_bytes;
} ds4_gpu_residual_set;

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const ds4_gpu_residual_set *residual,
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
        uint32_t                layer_index);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const ds4_gpu_residual_set *residual,
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
        uint32_t                slot_start,   /* TP Phase-3 batch split: owned slot range */
        uint32_t                slot_count,   /* 0 or n_expert => no split (full) */
        bool                   *mid_is_f16);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
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
        float                   eps);

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
        float                   norm_eps);

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* corr_delta (nullable): when non-NULL the kernel consumes routed[d]+delta[d]
 * — the go1b corr store-variant output — with the exact fadd the in-place corr
 * kernel used, keeping results bit-identical without a routed_out write hazard. */
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
        const ds4_gpu_tensor *corr_delta,
        uint32_t                n_embd,
        uint32_t                n_hc);

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
        uint32_t                n_hc);

/* go-onebit DQZ2 zchain (multiplicative correction chain; math contract in
 * ds4_zchain.h). ds4_gpu_zchain_set() uploads the whole packed table once at
 * load into small resident buffers:
 *   ops       [n_ops_total][16] f32, layer-major. Slot layout per op:
 *             [0]=type (1 GL | 2 dyn2 | 3 dyn8 | 4 TREF), [1]=g or t,
 *             [2..5]=w2p, [6..14]=w8, [15]=dyn8 V8 block index (-1 = none).
 *   layer_off [n_layer+1] op range per layer (ops[layer_off[l]..layer_off[l+1]))
 *   v8        concatenated fp16 [n_v8_blocks][8][d_model] dyn8 projections
 *   ge        [n_layer][n_expert] f32 router-weight gains (1.0-filled rows for
 *             layers without GE)
 *   ge_present[n_layer] flags so no-GE layers skip the dispatch entirely.
 *
 * ds4_gpu_zchain_ge_apply(): weights[t][k] *= ge[layer][selected[t][k]] --
 * BEFORE the routed matvec and before any compact-slot remap (original ids).
 * ds4_gpu_zchain_scale_routed(): routed[t][:] *= λ_layer(x[t]) with λ folded
 * in-kernel over the layer's ops (feature reductions on x per token) -- AFTER
 * the routed accumulate (and any TP all-reduce), BEFORE the additive corr.
 * Both return 1 on success (including the layer-has-nothing fast path). */
int ds4_gpu_zchain_set(
        const float        *ops,
        const uint32_t     *layer_off,
        const uint16_t     *v8,
        const float        *ge,
        const uint8_t      *ge_present,
        uint32_t             n_layer,
        uint32_t             n_expert,
        uint32_t             d_model,
        uint32_t             n_ops_total,
        uint32_t             n_v8_blocks);

int ds4_gpu_zchain_ge_apply(
        ds4_gpu_tensor       *weights,
        const ds4_gpu_tensor *selected,
        uint32_t                layer,
        uint32_t                n_expert_used,
        uint32_t                n_tokens);

int ds4_gpu_zchain_scale_routed(
        ds4_gpu_tensor       *routed,
        const ds4_gpu_tensor *x,
        uint32_t                layer,
        uint32_t                n_tokens);

/* frozen z^L (type 6, 2026-07-14): upload packed fp16 factors (concat of
 * z[k]|U[d*k]|V[d*k] per zl layer) + per-layer {offset-in-halves, rank, trust
 * factor}. The scale_routed dispatch applies it after the λ scale. k[l]==0 for
 * every layer (or n_layer==0) = nothing to do, returns 1. */
int ds4_gpu_zchain_zl_set(
        const uint16_t *zlm,
        const uint32_t *off,
        const uint32_t *k,
        const uint32_t *din,    /* md86: per-layer V input dim (NULL = all d_model; 3d = ftA) */
        const float    *tr,     /* type6: trust cap | type7 AMP: tanh 定标 scale */
        const uint32_t *mul,    /* per-layer 1=乘性AMP(type7, 2026-08-19), NULL=全加性 */
        uint32_t         n_layer,
        uint64_t         total_halves);

/* 路由闭式侧车(type8 zl.RTE, 2026-08-19): blob = z[k]|U[n_expert*k]|V[d_model*k]
 * per rte layer; 应用 δlogits = U·diag(z)·tanh(Vᵀx/s) 加在 router raw logits 上
 * (select 前)。k[l]==0 = 该层无侧车。 */
int ds4_gpu_zchain_rte_set(
        const uint16_t *rm,
        const uint32_t *off,
        const uint32_t *k,
        const float    *scale,
        uint32_t         n_layer,
        uint32_t         n_expert,
        uint64_t         total_halves);

/* logits[n_tokens][n_expert] += route 侧车 δ(x); 无侧车层零成本返回 1。 */
int ds4_gpu_zchain_route_bias(
        ds4_gpu_tensor       *logits,
        const ds4_gpu_tensor *x,
        uint32_t               layer,
        uint32_t               n_tokens);

#endif
