/* core_gpu_graph.h — GPU 图状态类型 + GPU 侧跨文件声明(整文件 DS4_NO_GPU 屏蔽)。
 * 只被 core_internal.h 包含。 */
#ifndef DS4_CORE_GPU_GRAPH_H
#define DS4_CORE_GPU_GRAPH_H
#ifndef DS4_NO_GPU

/*
 * Apple Metal stores the persistent attention-compressed KV cache in F16.  The
 * compressor still pools, normalizes, RoPEs, and FP8-rounds rows in F32 staging
 * before writing the cache, while checkpoints and debug dumps expand back to
 * F32 for the stable external format.  This is a storage optimization rather
 * than a semantic approximation: all Metal attention consumers already run the
 * compressed K/V rows through F16 FlashAttention/indexed-attention paths.
 */
#if defined(__APPLE__)
#define DS4_GPU_ATTN_COMP_CACHE_F16 1
#else
#define DS4_GPU_ATTN_COMP_CACHE_F16 0
#endif

/* =========================================================================
 * Metal Release Graph State.
 * =========================================================================
 *
 * The release Metal executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

typedef struct {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    ds4_gpu_tensor *cur_hc;
    ds4_gpu_tensor *flat_hc;
    ds4_gpu_tensor *hc_mix;
    ds4_gpu_tensor *hc_split;
    ds4_gpu_tensor *hc_pre;
    ds4_gpu_tensor *hc_post;
    ds4_gpu_tensor *hc_comb;
    ds4_gpu_tensor *attn_cur;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *qr_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_MAX_LAYER];
    uint32_t active_layer_start;
    uint32_t active_layer_end;
    bool active_layer_slice;

    /* Speculative decoding scratch.  MTP is allowed to mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  The prefix1 buffers are the cheap partial-accept state for the
     * common N=2 case. */
    ds4_gpu_tensor *spec_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_MAX_LAYER];
    uint32_t layer_n_index_comp[DS4_MAX_LAYER];
    uint32_t spec_prefix1_n_comp[DS4_MAX_LAYER];
    uint32_t spec_prefix1_n_index_comp[DS4_MAX_LAYER];
    /* spec replay 消除(2026-08-20): verify 批各层压缩器/indexer 输入行缓存(≤8 行),
     * restore 后用缓存行快进 acc 位 —— 免第二次全模型前向(实测 replay 55-65ms/轮)。 */
    ds4_gpu_tensor *spec_comp_rows_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_comp_rows_sc[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_idx_rows_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_idx_rows_sc[DS4_MAX_LAYER];
    int spec_comp_capture;
    bool spec_capture_prefix1;
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint32_t attn_comp_stage_cap;

    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    /* Side-stream private cur pair for the attn compressor chain: the main
     * stream's indexer compressor writes comp_kv_cur/comp_sc_cur concurrently,
     * so the side chain must not share them (2026-08-19 determinism fix). */
    ds4_gpu_tensor *comp_kv_side;
    ds4_gpu_tensor *comp_sc_side;
    ds4_gpu_tensor *attn_comp_stage;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *comp_mask;
    ds4_gpu_tensor *comp_selected;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_low;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn_hc;
    ds4_gpu_tensor *ffn_cur;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    /* go1b corr store-variant output [n_embd]: decode writes the correction
     * here (never into routed_out) so the tiny corr dispatch carries no write
     * hazard on the hot buffer; the fused shared-down consumer adds it. */
    ds4_gpu_tensor *corr_delta;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *after_ffn_hc;
    ds4_gpu_tensor *output_pre;
    ds4_gpu_tensor *output_weights;
    ds4_gpu_tensor *output_embd;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;

    /* Optional MTP model state.  It has its own raw cache because the drafter
     * runs on speculative future tokens; target KV state is updated only after
     * verification accepts draft tokens. */
    /* DSpark drafter(2026-08-18): target 层(40/41/42) HC 均值拼接 buffer, decode 单
     * token [3*4096]。capture 常开(node 开销 ~3μs), drafter ready 时被消费。 */
    ds4_gpu_tensor *dspark_main_hidden;
    ds4_gpu_tensor *dspark_main_x_raw;
    ds4_gpu_tensor *dspark_main_x;
    ds4_gpu_tensor *dspark_kv_tmp;
    ds4_gpu_tensor *dspark_win_kv[3];  /* 每块层环形窗 [128][512] f32 */
    ds4_gpu_tensor *dspark_ids;
    ds4_gpu_tensor *dspark_hc_pre;
    ds4_gpu_tensor *dspark_hc_w;
    ds4_gpu_tensor *dspark_flat;
    ds4_gpu_tensor *dspark_flat_norm;
    ds4_gpu_tensor *dspark_logits;
    ds4_gpu_tensor *spec_raw_save[DS4_MAX_LAYER]; /* verify 批写 SWA 环前的旧行存档 */
    ds4_gpu_tensor *dspark_spec_kv[3]; /* verify 批的 drafter KV 暂存 [BLK+1, HEAD_DIM] */
    ds4_gpu_tensor *dspark_conf;       /* [BLK] 逐位置置信(调度器输入) */
    ds4_gpu_tensor *dspark_prev_ids;   /* [BLK] 逐位置输入 token(置信头用) */
    ds4_gpu_tensor *dspark_prev_id;
    ds4_gpu_tensor *dspark_out_id;
    ds4_gpu_tensor *dspark_pf_hidden;   /* prefill 建窗: [chunk_cap, 3*4096] */
    ds4_gpu_tensor *dspark_pf_x;        /* [chunk_cap, 4096] */
    ds4_gpu_tensor *dspark_pf_kv;       /* [chunk_cap, 512] */
    int dspark_capture;               /* engine open 时按 drafter ready 置位 */
    ds4_gpu_tensor *mtp_embed;
    ds4_gpu_tensor *mtp_enorm;
    ds4_gpu_tensor *mtp_eproj;
    ds4_gpu_tensor *mtp_eproj_hc;
    ds4_gpu_tensor *mtp_hnorm_hc;
    ds4_gpu_tensor *mtp_hproj_hc;
    ds4_gpu_tensor *mtp_input_hc;
    ds4_gpu_tensor *mtp_state_hc;
    ds4_gpu_tensor *mtp_next_hc;
    ds4_gpu_tensor *mtp_raw_cache;
    uint32_t mtp_n_raw;
    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur_hc;
    ds4_gpu_tensor *batch_next_hc;
    ds4_gpu_tensor *batch_flat_hc;
    ds4_gpu_tensor *batch_hc_mix;
    ds4_gpu_tensor *batch_hc_split;
    ds4_gpu_tensor *batch_attn_cur;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_qr;
    ds4_gpu_tensor *batch_qr_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv;
    ds4_gpu_tensor *batch_comp_kv;
    ds4_gpu_tensor *batch_comp_sc;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_low;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_group_tmp;
    ds4_gpu_tensor *batch_low_tmp;
    ds4_gpu_tensor *batch_after_attn_hc;
    ds4_gpu_tensor *batch_ffn_cur;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_shared_gate;
    ds4_gpu_tensor *batch_shared_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_shared_out;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_mid;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_routed_out;
    bool batch_routed_mid_is_f16;
    ds4_gpu_tensor *batch_ffn_out;
    bool materialize_ffn_out;
    ds4_gpu_tensor *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    uint32_t power_percent;
    double prefill_layer_avg_sec[DS4_MAX_LAYER];
    double decode_token_avg_sec;
    bool quality;
    bool mtp_enabled;
    /* Tensor parallelism (Stage 2 skeleton). When tp != NULL the routed-MoE
     * output of decode layers [0, tp_layers) is recombined across two peers via
     * an all-reduce: after the (replicated) routed_moe each peer zeros the half
     * of routed_out it does not own and sum-all-reduces, reproducing the
     * single-machine value bit-for-bit before the residual combine. This minimal
     * element-range partition proves the split point + all-reduce frame + cross-
     * machine sync with no Metal kernel change; #06 replaces it with a real
     * compute split (masked experts / ff-row slice). tp == NULL ⇒ byte-identical
     * to the non-TP path. tp_vec is the host staging buffer for the exchange. */
    ds4_dist_tp *tp;
    uint32_t tp_layers;
    bool tp_owns_low; /* true ⇒ this peer owns the low half of routed_out */
    float *tp_vec;    /* host staging buffer [DS4_N_EMBD], allocated when tp set */
} ds4_gpu_graph;


typedef struct {
    int state;                       /* 0 未初始化 / 1 开 / -1 关 */
    int route_on;
    const float   *fin;              /* [NL][S][DIM] */
    const int32_t *ridx;             /* [NL][S][NACT] */
    const float   *rw;
    uint32_t S, nl, nact, dim;
    ds4_gpu_tensor *xbuf[DS4_MAX_LAYER];  /* per-layer: 同 buf 跨层复用会被未执行的
                                           * 前层 zchain kernel 读到后层数据 */
} ds4_ampanc_state;

typedef struct {
    float *gate_up_sum2;   /* [active layer][active expert][hidden] */
    float *down_sum2;      /* [active layer][active expert][expert FFN] */
    uint32_t gate_up_count[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t down_count[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    float *ffn_norm_buf;
    float *routed_mid_buf;
    uint16_t *routed_mid_f16_buf;
    int   *selected_buf;
    float *sq_tmp;
    uint32_t cap_tokens;
    uint64_t observed_tokens;
    uint64_t observed_routes;
    uint32_t chunks;
    const char *dataset_path;
} ds4_imatrix_collector;

extern ds4_ampanc_state g_ampanc;

/* ---- GPU 跨文件函数声明(拆分工序新增; 定义散于 core_gpu_*.c) ---- */
extern const char *g_dump_tag;
bool metal_graph_spec_raw_snapshot(ds4_gpu_graph *g, uint32_t il, uint32_t pos0, uint32_t n);
bool metal_graph_spec_raw_restore(ds4_gpu_graph *g, uint32_t pos0, uint32_t from, uint32_t to);
bool metal_graph_indexer_stage_profile_boundary( const char *stage, uint32_t il, uint32_t pos0, uint32_t n_tokens, uint32_t n_comp, double *stage_t0);
bool metal_graph_layer_stage_profile_boundary( const char *part, const char *stage, uint32_t il, uint32_t pos0, uint32_t n_tokens, double *stage_t0);
bool metal_graph_matmul_plain_tensor( ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *w, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
uint64_t argmax_f32(const float *x, uint64_t n);
int attn_output_kq_batch(const ds4_tensor *a, ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *map, uint64_t msize, uint64_t offb, uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens);
int dense_matmul_pair_typed(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, const ds4_model *m, const ds4_tensor *w0, const ds4_tensor *w1, uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim, const ds4_gpu_tensor *x);
int dense_matmul_typed(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
void eval_hdump_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens);
void forward_token_raw_swa_cpu( float * logits, const ds4_model * model, const ds4_weights * weights, ds4_kv_cache * cache, int token, uint32_t pos);
int generate_metal_graph_raw_swa( const ds4_model * model, const ds4_vocab * vocab, const ds4_weights * weights, const token_vec * prompt, int n_predict, int ctx_size, bool quality, int power_percent, const char * directional_steering_file, float directional_steering_attn, float directional_steering_ffn, ds4_token_emit_fn emit, ds4_generation_done_fn done, void * emit_ud, ds4_session_progress_fn progress, void * progress_ud);
void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec);
void graph_power_note_prefill_layer(ds4_gpu_graph *g, uint32_t il, double elapsed_sec);
bool graph_power_throttle_enabled(const ds4_gpu_graph *g);
bool imatrix_collect_layer_batch( ds4_imatrix_collector *c, ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens);
void imatrix_collector_free(ds4_imatrix_collector *c);
bool imatrix_collector_init(ds4_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path);
bool imatrix_collector_save( const ds4_imatrix_collector *c, const ds4_weights *weights, const char *path);
float max_abs_diff(const float *a, const float *b, uint64_t n);
ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes);
bool metal_graph_alloc_raw_cap( ds4_gpu_graph *g, const ds4_weights *weights, const ds4_layer_weights *layer, uint32_t raw_cap, uint32_t ctx_size, uint32_t prefill_cap, bool enable_mtp, uint32_t active_layer_start, uint32_t active_layer_end, bool active_layer_slice);
ds4_gpu_tensor *metal_graph_attn_comp_row_view( ds4_gpu_graph *g, uint32_t il, uint32_t row);
uint32_t metal_graph_attn_comp_update_row(uint32_t row);
ds4_gpu_tensor *metal_graph_attn_comp_update_target( ds4_gpu_graph *g, uint32_t il);
bool metal_graph_commit_attn_comp_stage( ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows);
uint64_t metal_graph_context_bytes_for_kv_policy( uint32_t ctx_size, uint32_t raw_cap, uint32_t prefill_cap, uint64_t *kv_cache_bytes_out);
void metal_graph_debug_dump_i32_tensor( const char *name, ds4_gpu_tensor *t, uint64_t n_i32, uint32_t il, uint32_t pos);
void metal_graph_debug_dump_tensor( const char *name, ds4_gpu_tensor *t, uint64_t n_f32, uint32_t il, uint32_t pos);
bool metal_graph_decode_hc_pre( ds4_gpu_tensor *out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const ds4_model *model, uint64_t scale_offset, uint64_t base_offset);
int metal_graph_decode_test( const ds4_model *model, const ds4_weights *weights, const token_vec *prompt);
bool metal_graph_direct_expert_read_enabled(void);
bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g);
bool metal_graph_dspark_state_restore(ds4_gpu_graph *g);
bool metal_graph_dspark_state_snapshot(ds4_gpu_graph *g);
bool metal_graph_dspark_step( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, const ds4_dspark_weights *dw, int anchor_token, uint32_t pos, int out_ids[DS4_DSPARK_BLK]);
bool metal_graph_dspark_step_n( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, const ds4_dspark_weights *dw, int anchor_token, uint32_t pos, int out_ids[DS4_DSPARK_BLK], uint32_t n_need, float *out_conf);
bool metal_graph_dspark_win_commit(ds4_gpu_graph *g, uint32_t pos0, uint32_t n_acc);
bool metal_graph_encode_decode_layer( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, uint32_t il, uint32_t pos, ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t raw_row, uint32_t n_raw, int token);
bool metal_graph_encode_layer_attention_batch( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, uint32_t il, uint32_t pos0, uint32_t n_tokens);
bool metal_graph_encode_layer_attention_batch_stages( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, uint32_t il, uint32_t pos0, uint32_t n_tokens, uint32_t stages);
bool metal_graph_encode_layer_batch( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, uint32_t il, uint32_t pos0, uint32_t n_tokens);
bool metal_graph_encode_layer_ffn_batch( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, uint32_t il, uint32_t pos0, uint32_t n_tokens);
bool metal_graph_encode_layer_ffn_batch_ex( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, uint32_t il, uint32_t pos0, uint32_t n_tokens, bool no_zchain);
bool metal_graph_encode_output_head( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, uint64_t vocab_dim);
bool metal_graph_encode_output_head_batch( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, uint32_t n_tokens, uint64_t vocab_dim);
bool metal_graph_encode_token_raw_swa( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, int token, uint32_t pos, bool need_logits, bool allow_split_flush);
bool metal_graph_eval_token_raw_swa( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, int token, uint32_t pos, float *logits);
int metal_graph_first_token_full_test( const ds4_model *model, const ds4_weights *weights, const token_vec *prompt);
void metal_graph_free(ds4_gpu_graph *g);
bool metal_graph_layer_is_active(const ds4_gpu_graph *g, uint32_t il);
bool metal_graph_load_directional_steering( ds4_gpu_graph *g, const char *path, float attn_scale, float ffn_scale);
bool metal_graph_matmul_q8_0_named_tensor( const char *module, uint32_t il, uint32_t pos0, ds4_gpu_tensor *out, const ds4_model *model, const ds4_tensor *w, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
uint32_t metal_graph_prefill_cap_for_prompt(int prompt_len);
bool metal_graph_prefill_chunked( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, const token_vec *prompt, int n_tokens, float *logits, bool show_progress, ds4_session_progress_fn progress, void *progress_ud, ds4_session_progress_fn display_progress, void *display_progress_ud);
bool metal_graph_prefill_chunked_range( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, const token_vec *prompt, uint32_t start, uint32_t n_tokens, float *logits, bool show_progress, ds4_session_progress_fn progress, void *progress_ud, ds4_session_progress_fn display_progress, void *display_progress_ud, ds4_imatrix_collector *imatrix);
bool metal_graph_prefill_layer_major( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, const token_vec *prompt, uint32_t start, uint32_t n_tokens, float *logits, bool show_progress, ds4_imatrix_collector *imatrix, ds4_session_progress_fn display_progress, void *display_progress_ud);
bool metal_graph_prefill_raw_swa( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, const token_vec *prompt, int n_tokens, float *logits, bool show_progress, ds4_session_progress_fn display_progress, void *display_progress_ud);
int metal_graph_prompt_logits_test( const ds4_model *model, const ds4_weights *weights, const token_vec *prompt, int ctx_size);
bool metal_graph_q_stage_profile_boundary( const char *stage, uint32_t il, uint32_t pos0, uint32_t n_tokens, double *stage_t0);
uint32_t metal_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap);
uint32_t metal_graph_raw_span_for_batch( const ds4_gpu_graph *g, uint32_t pos0, uint32_t n_tokens);
bool metal_graph_reset_prefill_state(ds4_gpu_graph *g);
uint32_t metal_graph_resume_prefill_min_tokens(void);
bool metal_graph_spec_comp_fastforward(ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, uint32_t pos0, uint32_t acc);
ds4_gpu_tensor *metal_graph_tensor_row_view( ds4_gpu_tensor *base, uint32_t row, uint64_t row_values);
uint32_t metal_graph_token_split_after_layers(void);
bool metal_graph_upload_prompt_embeddings_hc( ds4_gpu_tensor *out_hc, ds4_gpu_tensor *tokens, const ds4_model *model, const ds4_weights *weights, const token_vec *prompt, uint32_t pos0, uint32_t n_tokens);
bool metal_graph_upload_prompt_tokens( ds4_gpu_tensor *out_tokens, const token_vec *prompt, uint32_t pos0, uint32_t n_tokens);
bool metal_graph_use_reference_hc_decode(void);
bool metal_graph_use_reference_hc_norm_decode(void);
bool metal_graph_use_reference_qkv_norm(void);
bool metal_graph_warmup_prefill_kernels( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, uint32_t n_tokens);
bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n);
int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes, uint8_t *buf, size_t cap, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor, uint64_t offset_f16, uint64_t count, uint8_t *buf, size_t cap, char *err, size_t errlen);
const ds4_gpu_residual_set *residual_set_for(const ds4_model *m, uint32_t il);
float rms_abs_diff(const float *a, const float *b, uint64_t n);
void router_freq_collect(ds4_gpu_tensor *logits_t, uint32_t il, uint32_t n_tokens);
uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len);
uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len);
void trace_hnorm_head(ds4_gpu_graph *g, uint32_t vocab_dim);
int zchain_gpu_upload(const struct ds4_zchain *z);
int ampanc_on(void);
void cap_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens);
void cap_decode_layer(ds4_gpu_graph *g, uint32_t il);
void engine_register_layer_routers(ds4_engine *e, uint32_t start, uint32_t end);
bool metal_graph_alloc( ds4_gpu_graph *g, const ds4_weights *weights, const ds4_layer_weights *layer);
bool metal_graph_apply_directional_steering_attn( ds4_gpu_graph *g, ds4_gpu_tensor *x, uint32_t il, uint32_t rows);
bool metal_graph_apply_directional_steering_ffn( ds4_gpu_graph *g, ds4_gpu_tensor *x, uint32_t il, uint32_t rows);
uint32_t metal_graph_attn_comp_cache_is_f16(void);
ds4_gpu_tensor *metal_graph_attn_comp_prefill_target( ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows);
void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t);
bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il);
bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il);
void metal_graph_debug_dump_f16_tensor( const char *name, ds4_gpu_tensor *t, uint64_t n_f16, uint32_t il, uint32_t pos);
uint32_t metal_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g);
bool metal_graph_decode_kv_store( ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t raw_row);
bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g);
bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g);
bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g);
bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos);
uint32_t metal_graph_raw_start_for_span( const ds4_gpu_graph *g, uint32_t last_pos, uint32_t n_raw);
bool metal_graph_refresh_ratio4_compressor_state( ds4_gpu_graph *g, const ds4_model *model, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_tensor *kv_weight, const ds4_tensor *score_weight, const ds4_tensor *ape, uint32_t head_dim, uint32_t width, uint32_t pos0, uint32_t n_tokens);
void metal_graph_trace_layer_stages( ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer, const float *cpu_in_hc, uint32_t il, int token);
bool metal_graph_use_reference_attn_out_hc(void);
bool metal_graph_use_reference_compressor_pair_proj(void);
bool metal_graph_use_reference_kv_decode(void);
bool metal_graph_use_reference_shared_down_hc(void);
int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes, uint8_t *buf, size_t cap, uint64_t *remaining, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor, uint64_t offset_f16, uint64_t count, uint8_t *buf, size_t cap, uint64_t *remaining, char *err, size_t errlen);
void trace_hnorm_layer(ds4_gpu_graph *g, uint32_t il, uint32_t pos, uint64_t hc_dim);

#endif /* !DS4_NO_GPU */
#endif /* DS4_CORE_GPU_GRAPH_H */
