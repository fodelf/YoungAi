/* core_types.h — ds4.c 内部共享类型(机械拆分, 与原文逐字一致)。
 * 只被 core_internal.h 包含; 不要单独 include。 */
#ifndef DS4_CORE_TYPES_H
#define DS4_CORE_TYPES_H


/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads:
 *   - Q2_K routed down experts
 *   - Q4_K routed experts in the high-memory variant
 *   - IQ2_XXS routed gate/up experts
 *   - Q8_K temporary activation blocks for dot products
 */
#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

/* go1b: DS4 strict-1-bit routed expert weight.  One fp16 per-row scale d
 * (replicated into every QK_K block of the row) + QK_K sign bits; element j in
 * a block is +d when signs[j/8] bit (j%8) is set, else -d.  34 bytes / 256
 * elements.  Mirrors the metal block_go1b in metal/moe.metal. */
typedef struct {
    uint16_t d;
    uint8_t  signs[QK_K / 8];
} block_go1b;

/* go2b: offline-merged base+residual strict-binary pair (R5-C). Element value
 * is the exact 4-level sum (±d1)+(±d2) of the two go1b layers it replaces.
 * 68 bytes / 256 elements. Mirrors metal block_go2b. */
typedef struct {
    uint16_t d1;
    uint16_t d2;
    uint8_t  s1[QK_K / 8];
    uint8_t  s2[QK_K / 8];
} block_go2b;

#define DS4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
DS4_STATIC_ASSERT(ds4_block_q2_k_size, sizeof(block_q2_K) == 84);
DS4_STATIC_ASSERT(ds4_block_q4_k_size, sizeof(block_q4_K) == 144);
DS4_STATIC_ASSERT(ds4_block_q8_k_size, sizeof(block_q8_K) == 292);
DS4_STATIC_ASSERT(ds4_block_iq2_xxs_size, sizeof(block_iq2_xxs) == 66);
DS4_STATIC_ASSERT(ds4_block_go1b_size, sizeof(block_go1b) == 34);
DS4_STATIC_ASSERT(ds4_block_go2b_size, sizeof(block_go2b) == 68);

typedef struct {
    uint32_t ctx_size;
    uint32_t comp_cap;
    uint32_t attn_score_cap;
    uint32_t q8_cap;

    float *plain;
    float *cur;
    float *next;

    float *attn_cur;
    float *attn_norm;
    float *attn_residual;
    float *q;
    float *qr;
    float *qr_norm;
    float *kv_raw;
    float *kv;
    float *heads;
    float *attn_low;
    float *attn_out;
    float *after_attn_hc;
    float *attn_score;

    float *comp;
    float *index_comp;
    float *comp_kv_cur;
    float *comp_sc_cur;
    float *comp_pooled;

    bool *index_allowed;
    float *index_q;
    float *index_weights;
    float *index_scores;

    float *ffn_cur;
    float *ffn_norm;
    float *ffn_moe;
    float *ffn_shared;
    float *ffn_out;
    float *shared_gate;
    float *shared_up;
    float *shared_mid;
    float *routed_mid_all;
    block_q8_K *routed_xq;
    block_q8_K *routed_midq;

    int8_t *q8_xq;
    float *q8_xscale;

    float *hc_flat;
    float *output_flat;
    float *output_pre;
    float *output_weights;
    float *output_embd;
    float *output_norm;
} ds4_cpu_decode_scratch;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;


typedef void (*ds4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);


enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;


typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;


typedef struct {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    /* V4.1 专属(V4 模型下全 NULL): indexer 键由压缩 latent 经 wk+k_norm 得(只在 kv 源层);
     * engram 三件只在 engram 层(表本体在盘, 见 g_ds4_v41)。 */
    ds4_tensor *indexer_wk;
    ds4_tensor *indexer_k_norm;
    ds4_tensor *engram_wkv;
    ds4_tensor *engram_q;
    ds4_tensor *engram_k;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
} ds4_layer_weights;

#include "core_draft_tower_types.h"

typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    /* V4.1 engram 哈希常量(转换器从 tokenizer 算好嵌入的张量; V4 下 NULL) */
    ds4_tensor *engram_token_map;     /* I32 [vocab]: 原 id → 压缩 id */
    ds4_tensor *engram_multipliers;   /* I64 [n_engram][max_ngram] */
    ds4_tensor *engram_primes;        /* I64 [n_engram][max_ngram-1][heads] */
    ds4_tensor *engram_offsets;       /* I64 [n_engram][(max_ngram-1)*heads] */
    ds4_layer_weights layer[DS4_MAX_LAYER];
    ds4_draft_tower_weights mtp;   /* DSpark 三塔(speed.md 段 6); 定义见 core_draft_tower_types.h */
} ds4_weights;

typedef struct {
    ds4_tensor *e_proj;
    ds4_tensor *h_proj;
    ds4_tensor *enorm;
    ds4_tensor *hnorm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_layer_weights block;
} ds4_mtp_weights;

/* 0731 官方 DSpark drafter(块并行 5-token, 共享主模型 embd/lm_head):
 * mtp.0..2 三个完整 V4 层 + main_proj(target-hidden 融合) + confidence/markov 头。
 * 张量由量化器 --mtp-append 注入主 GGUF(v3+); 缺席 ⇒ ready=false, 纯解码不受扰。 */
#define DS4_DSPARK_MAX_BLOCKS 3
typedef struct {
    ds4_tensor *main_proj;
    ds4_tensor *main_norm;
    ds4_tensor *confidence_proj;   /* 可缺(置信门) */
    ds4_tensor *markov_w1;         /* 可缺(块内半自回归头) */
    ds4_tensor *markov_w2;
    ds4_tensor *norm;              /* 出口 norm */
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_layer_weights block[DS4_DSPARK_MAX_BLOCKS];
    int n_blocks;
    bool ready;
    const ds4_model *src;      /* drafter 张量所属 model(主 model 或 DS4_DRAFT_GGUF 副文件) */
    const ds4_model *head_src; /* 出口侧(norm/hc_head_*)张量所属 model: 官方语义 drafter 复用
                                * 主模型 head(self.mtp[-1].head = self.head), checkpoint 的
                                * mtp.* 命名空间不含这些 ⇒ 独立 drafter 文件时从主模型借 */
} ds4_dspark_weights;


typedef struct {
    uint64_t off;
    uint64_t end;
} ds4_model_map_span;

typedef struct {
    ds4_model_map_span *v;
    uint32_t len;
    uint32_t cap;
    uint64_t max_tensor_bytes;
} ds4_model_map_span_vec;

typedef struct {
    float *out;
    const uint16_t *data;
    const float *x;
    uint64_t in_dim;
} matvec_f16_ctx;


typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t row0;
    uint64_t blocks;
} matvec_q8_0_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} matvec_q8_0_pair_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
    uint64_t rank;
} matvec_q8_0_grouped_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t n_groups;
    uint64_t group_dim;
    uint64_t blocks;
    uint64_t rank;
} matmul_q8_0_grouped_batch_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_pair_batch_ctx;

typedef struct {
    const float *x;
    int8_t *xq;
    float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} quantize_q8_0_batch_ctx;

typedef struct {
    uint32_t token;
    uint32_t slot;
} ds4_expert_pair;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_iq2_xxs_batch_mid_ctx;

typedef struct {
    const float *mid;
    block_q8_K *midq;
    uint64_t down_in_dim;
    uint64_t down_blocks;
} quantize_mid_pairs_ctx;

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_accum_rows_ctx;

typedef struct {
    float *raw_kv;
    uint32_t n_raw;
    uint32_t cap_raw;

    uint32_t compress_ratio;
    uint32_t comp_cap;
    uint32_t n_comp;
    float *attn_comp_kv;
    float *attn_state_kv;
    float *attn_state_score;

    uint32_t n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} ds4_layer_cache;

typedef struct {
    ds4_layer_cache layer[DS4_MAX_LAYER];
    uint32_t head_dim;
} ds4_kv_cache;



typedef struct ds4_vocab ds4_vocab;


typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;


struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int user_id;
    int assistant_id;
    int system_id;        /* <｜System｜>: V4.1 的 tokenizer 才有(id 128799), 没有 = -1, 聊天渲染就不写它(V4 的模板本来没有) */
    int think_start_id;
    int think_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

struct ds4_engine {
    ds4_model model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_mtp_weights mtp_weights;
    ds4_dspark_weights dspark;    /* 0731 官方块并行 drafter(v3+ GGUF 内嵌) */
    ds4_backend backend;
    int mtp_draft_tokens;
    float mtp_margin;
    /* Go trie drafter for copy-spec (--go-trie / DS4_GO_TRIE); NULL = off. */
    /* knowledge-MTP reference corpus (ds4_mtp.c: built-in idioms +
     * DS4_REF_CORPUS extension files). Always built at init; NULL only on
     * OOM, and every module entry point accepts NULL as "no corpus". */
    /* 多模态 registry (ds4_multimodal.c): modality -> text/tokens. The
     * "image" family auto-binds the frontend-domain UI-sketch encoder at
     * open (DS4_MM_IMAGE_CMD env > ./mm-ui). NULL only on OOM; consumers
     * fail closed on NULL rather than fake support. */
    ds4_mm *mm;
    char *directional_steering_file;
    float *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    int power_percent;
    bool quality;
    ds4_distributed_options distributed;
    bool metal_ready;
    /* TP peer connection (Stage 2). Established lazily on first session use when
     * distributed.tp_enabled, shared read-only by the session graph. */
    ds4_dist_tp *tp;
    bool tp_owns_low; /* coordinator owns low expert positions */
};

#endif /* DS4_CORE_TYPES_H */
