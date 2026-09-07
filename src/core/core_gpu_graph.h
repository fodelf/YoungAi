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
/* 压缩缓存(attn comp 行)与 indexer 缓存两后端一律 f16, 无开关(09-07)。 */
/* indexer 压缩缓存(2026-09-06, 1M 战役): 两个后端一律 f16, 没有开关。行值是 hadamard+FP4 QAT 后的量化格点
 * (f32 里存的就是 scale×小整数), f16 存几乎无损; 写入一律"f32 暂存(attn_comp_stage) → QAT → 提交转 f16"
 * (core_gpu_decode_util.c), 快照/调试导出仍按 f32 外部格式。曾做成 Apple 0/CUDA 1 的开关: 暂存分配与容量挂在
 * 另一个开关下, 两开关不同步 ⇒ CUDA 运行时炸两次(09-06), 故拔掉。 */

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
    /* 09-07 小批压缩器改走解码攒行机制: 轮前快照要带上环里攒着的行 + 计数(comp_x_pending/last_pos), 部分接受后
     * 恢复再按接受位重放 push/emit。spec_ring_save[il] = ratio 行 × DS4_N_EMBD f32(懒分配, 非 capture 时机)。 */
    ds4_gpu_tensor *spec_ring_save[DS4_MAX_LAYER];
    uint32_t spec_prefix1_x_pending[DS4_MAX_LAYER];
    uint32_t spec_prefix1_x_last[DS4_MAX_LAYER];
    /* 按需快照(09-07): 批内 emit 位 t_e(无则 -1)。压缩器 state 只在 emit 位变, 所以只有 t_e ≥ 1 的层要拷 state/环/行;
     * 回滚只在 t_e ≥ acc(emit 用了被拒行)时做, 其余层只改计数。此前每轮 43 层全拷 = 500 次 memcpy/轮。 */
    int8_t   spec_emit_t[DS4_MAX_LAYER];
    uint32_t spec_pos0, spec_k;
    ds4_gpu_tensor *spec_heads_alt;   /* 小批跨 indexer 门槛轮的稠密注意力暂存(8 行, 懒分配) */
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
    /* 压缩器投影攒批(2026-09-05, core_gpu_decode_comp.c): 每压缩层一个 x 行环(ratio 行×n_embd),
     * 解码每 token 只推一行, emit 时一次算攒下的 n 行投影进 comp_kv_batch/comp_sc_batch
     * (128 行×最大压缩宽)。pending = 环上攒着还没入 state 的行数, last_pos = 最后推入的 pos。 */
    ds4_gpu_tensor *comp_x_ring[DS4_MAX_LAYER];
    uint32_t comp_x_pending[DS4_MAX_LAYER];
    uint32_t comp_x_last_pos[DS4_MAX_LAYER];
    ds4_gpu_tensor *comp_kv_batch;
    ds4_gpu_tensor *comp_sc_batch;
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
    /* 预发射(2026-09-07, core_gpu_imatrix.c metal_graph_eval_token_raw_swa): 贪心解码把 pos+1 的图在
     * pos 图跑完前排进流, token 由图末尾的设备 argmax 供给。只在 pos+1 非 emit 位做(设备侧只写可覆写
     * 行), 主机计数器留快照, 调用方喂的 token 与设备 argmax 不一致时回滚重编码。 */
    float *logits_pinned;        /* 本图 logits 异步回传落点(pinned, DS4_N_VOCAB) */
    int32_t *tok_next_pinned;    /* 本图设备 argmax 回传落点 */
    int prelaunch_capable;       /* 0 未查, 1 能(CUDA), -1 不能(Metal) */
    int prelaunch_want;          /* 会话声明: 下一 token = 无惩罚 argmax(core_session_sample.c) */
    int argmax_exclude;          /* argmax 排除的 id(-1 无; ds4-bench 排除 EOS 保持续写) */
    int64_t prelaunched_pos;     /* 已预发射的位置(-1 无) */
    int32_t prelaunched_token;   /* 预发射图消费的 token = 上一图 logits 的设备 argmax */
    /* 已预捕获(未必预发射)的位置(-1 无): 预捕获 = 在捕获态 encode pos+1, 主机计数器(layer_n_comp/
     * comp_x_pending)已按 pos+1 推进。下一步若不是 eval(pos+1)(投机 verify 批/回退), 必须先回滚到
     * snap_* 并作废待发射图(metal_graph_token_pending_discard), 否则 pos+1 被推进两次(09-07 定罪:
     * --spec 首跑 "cuda decode failed")。 */
    int64_t pend_pos;
    uint32_t snap_n_comp[DS4_MAX_LAYER];        /* 预捕获前的主机计数器快照(对账失败时回滚) */
    uint32_t snap_n_index_comp[DS4_MAX_LAYER];
    uint32_t snap_x_pending[DS4_MAX_LAYER];
    /* 09-07 定罪: 预编码 pos+1 的 comp_push 把 comp_x_last_pos 推到 pos+1, 回滚漏了它 ⇒ 投机第 1 轮快照按陈旧 last_pos 算
     * 攒行环保存段(错一格), 部分接受回滚把错位段写回环, ratio-4 层那块压缩行被污染(2.8K 意语第 377 字节分叉, pos0%4==2/3 相位才撞)。 */
    uint32_t snap_x_last[DS4_MAX_LAYER];
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
void eval_hdump_tensor_rows(const ds4_gpu_tensor *hc, uint32_t il, uint32_t n_tokens);
void eval_hdump_raw_rows(const ds4_gpu_tensor *raw, uint32_t tag, uint32_t slot0, uint32_t n, uint32_t raw_cap);   /* --eval-hdump 通用行写出(解码路也用) */
void eval_hdump_pos(uint32_t pos0, uint32_t n);                 /* --eval-hdump 位置边车 h_pos.bin(与 L99 同序, 脚本按位置对齐) */
void eval_hdump_logits_rows(const float *logits, uint32_t n);   /* --eval-hdump 主机侧 logits 行 h_L96.bin(与 L99 同序) */
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
/* 两种缓存(压缩行 / indexer 行)恒 f16: 写入方把 f32 行写进 g->attn_comp_stage 行 0 起并做量化, 然后由 commit 抄成 f16 落缓存 */
bool metal_graph_commit_attn_comp_stage( ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows);
bool metal_graph_commit_index_comp_stage(ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows);
/* 压缩器投影攒批(core_gpu_decode_comp.c): push 每 token 一行; project_pending 在 emit 位一次算
 * 攒下的 n 行到 comp_kv_batch/comp_sc_batch; flush_pending 是非解码入口(prefill/快照/回卷)
 * 前的强制冲刷(只入 state 不池化); pending_clear 在 state 被整体重建/重载后调。 */
bool metal_graph_comp_push(ds4_gpu_graph *g, uint32_t il, uint32_t ratio, uint32_t pos);
bool metal_graph_comp_push_row(ds4_gpu_graph *g, uint32_t il, uint32_t ratio, uint32_t pos, const ds4_gpu_tensor *row);
bool metal_graph_comp_emit_step(ds4_gpu_graph *g, const ds4_model *model, const ds4_layer_weights *layer,
                                uint32_t il, uint32_t ratio, bool compressed, float freq_base, float freq_scale,
                                float ext_factor, float attn_factor);
bool metal_graph_comp_project_pending( ds4_gpu_graph *g, const ds4_model *model, const ds4_tensor *kv_weight, const ds4_tensor *gate_weight, uint32_t il, uint32_t ratio, uint32_t width, uint32_t *n_out, uint32_t *pos0_out);
bool metal_graph_comp_flush_pending(ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights);
void metal_graph_comp_pending_clear(ds4_gpu_graph *g);
uint64_t metal_graph_context_bytes_for_kv_policy( uint32_t ctx_size, uint32_t raw_cap, uint32_t prefill_cap, uint64_t *kv_cache_bytes_out);
bool metal_graph_decode_hc_pre( ds4_gpu_tensor *out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const ds4_model *model, uint64_t scale_offset, uint64_t base_offset);
int metal_graph_decode_test( const ds4_model *model, const ds4_weights *weights, const token_vec *prompt);
bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g);
bool metal_graph_dspark_state_restore(ds4_gpu_graph *g, uint32_t acc);
bool metal_graph_dspark_state_snapshot(ds4_gpu_graph *g, uint32_t pos0, uint32_t k);
void metal_graph_token_pending_discard(ds4_gpu_graph *g);   /* 预捕获图作废 + 主机计数器回滚(见 pend_pos) */
void metal_graph_token_pending_forget(ds4_gpu_graph *g);    /* 只作废不回滚: 计数器已被外部整体改写时用 */
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
bool metal_graph_warmup_prefill_kernels( ds4_gpu_graph *g, const ds4_model *model, const ds4_weights *weights, uint32_t n_tokens);
bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n);
int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes, uint8_t *buf, size_t cap, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor, uint64_t offset_f16, uint64_t count, uint8_t *buf, size_t cap, char *err, size_t errlen);
const ds4_gpu_residual_set *residual_set_for(const ds4_model *m, uint32_t il);
float rms_abs_diff(const float *a, const float *b, uint64_t n);
uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len);
uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len);
int zchain_gpu_upload(const struct ds4_zchain *z);
int ampanc_on(void);
void cap_batch_layer(ds4_gpu_graph *g, uint32_t il, uint32_t n_tokens);
void cap_decode_layer(ds4_gpu_graph *g, uint32_t il);
void engine_register_layer_routers(ds4_engine *e, uint32_t start, uint32_t end);
bool metal_graph_alloc( ds4_gpu_graph *g, const ds4_weights *weights, const ds4_layer_weights *layer);
bool metal_graph_apply_directional_steering_attn( ds4_gpu_graph *g, ds4_gpu_tensor *x, uint32_t il, uint32_t rows);
bool metal_graph_apply_directional_steering_ffn( ds4_gpu_graph *g, ds4_gpu_tensor *x, uint32_t il, uint32_t rows);
bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il);
bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il);
uint32_t metal_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g);
bool metal_graph_decode_kv_store( ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t raw_row);
bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g);
bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g);
bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g);
bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos);
uint32_t metal_graph_raw_start_for_span( const ds4_gpu_graph *g, uint32_t last_pos, uint32_t n_raw);
bool metal_graph_refresh_ratio4_compressor_state( ds4_gpu_graph *g, const ds4_model *model, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_tensor *kv_weight, const ds4_tensor *score_weight, const ds4_tensor *ape, uint32_t head_dim, uint32_t width, uint32_t pos0, uint32_t n_tokens);
int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor, uint64_t offset, uint64_t bytes, uint8_t *buf, size_t cap, uint64_t *remaining, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor, uint64_t offset_f16, uint64_t count, uint8_t *buf, size_t cap, uint64_t *remaining, char *err, size_t errlen);
/* 压缩缓存行格式 ↔ 外部 f32 行(core_payload.c) */
DS4_MAYBE_UNUSED int payload_write_comp_rows_as_f32(FILE *fp, const ds4_gpu_tensor *tensor, uint64_t rows, uint8_t *buf, size_t cap, char *err, size_t errlen);
DS4_MAYBE_UNUSED int payload_read_f32_as_comp_rows(FILE *fp, ds4_gpu_tensor *tensor, uint64_t rows, uint8_t *buf, size_t cap, uint64_t *remaining, char *err, size_t errlen);

#endif /* !DS4_NO_GPU */
#endif /* DS4_CORE_GPU_GRAPH_H */
