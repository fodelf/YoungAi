/* ds4_internal.h — shared internal types for the DS4 runtime modules.
 *
 * R3-g modularization (user directive: maintainability over the old
 * single-file style): the model/tensor/shape/corr type web moved here so
 * subsystem modules (ds4_corr.c first, capture/mtp next) can live in their
 * own translation units. Everything here is engine-internal — the public
 * API stays in ds4.h. Pure code motion from ds4.c; no behavior change. */
#ifndef DS4_INTERNAL_H
#define DS4_INTERNAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "ds4.h"
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

enum {
    DS4_MAX_LAYER            = 61,
    DS4_MAX_EMBD             = 7168,
    DS4_MAX_VOCAB            = 129280,
    DS4_MAX_HEAD             = 128,
    DS4_MAX_HEAD_KV          = 1,
    DS4_MAX_HEAD_DIM         = 512,
    DS4_MAX_VALUE_DIM        = 512,
    DS4_MAX_ROT              = 64,
    DS4_MAX_OUT_GROUP        = 16,
    DS4_MAX_LORA_Q           = 1536,
    DS4_MAX_LORA_O           = 1024,
    DS4_MAX_EXPERT           = 384,
    DS4_MAX_EXPERT_USED      = 6,
    DS4_MAX_EXPERT_SHARED    = 1,
    DS4_MAX_FF_EXP           = 3072,
    DS4_MAX_HASH_LAYER       = 3,
    DS4_MAX_SWA              = 128,
    DS4_MAX_INDEXER_HEAD     = 64,
    DS4_MAX_INDEXER_HEAD_DIM = 128,
    DS4_MAX_INDEXER_TOP_K    = 1024,
    DS4_MAX_HC               = 4,
    DS4_MAX_HC_SINKHORN_ITER = 20,
};

typedef enum {
    DS4_VARIANT_FLASH = 0,
    DS4_VARIANT_PRO   = 1,
    DS4_VARIANT_V41   = 2,   /* DeepSeek V4.1 Flash(2026-09-12 战役): 40 层 CED, 共享压缩 KV 源层, engram */
} ds4_variant;

/* V4.1 专属结构元数据(全部来自 GGUF deepseek4.* 键, 由 core_validate_v41.c 装填; V4 模型下 active=0)。
 * 为什么单列一个结构而不塞进 ds4_shape: shape 是"维度", 这些是"接线"(哪层读哪层的缓存),
 * 热路径按层查表, 不能每层再去扫元数据。 */
#define DS4_V41_MAX_ENGRAM 4
#define DS4_MTP_MAX_TOWERS 4      /* DSpark 草稿塔数上限(官方 3); 真值读元数据 */
#define DS4_MTP_MAX_EXPERTS 256   /* 每塔专家数上限(官方 128); 真值读元数据 */
typedef struct {
    int      active;
    uint8_t  is_kv_source[DS4_MAX_LAYER];     /* 本层自己压缩并持有压缩 KV 缓存 */
    int16_t  kv_source_of[DS4_MAX_LAYER];     /* 本层读哪层的压缩 KV(源层=自己; ratio 0 层 = -1) */
    uint8_t  is_index_source[DS4_MAX_LAYER];  /* 本层自己跑 indexer 产 topk */
    int16_t  index_source_of[DS4_MAX_LAYER];  /* 本层用哪层的 topk(ratio 0 层 = -1) */
    int32_t  candidate_source_layer;          /* 两级 topk: 该层筛候选块, 其后各层只在块内 topk; <0 = 无 */
    int32_t  candidate_topk_blocks, candidate_block_size;
    uint32_t n_engram;
    int32_t  engram_layer[DS4_V41_MAX_ENGRAM];
    int16_t  engram_index_of[DS4_MAX_LAYER];  /* 层 → engram 序号, -1 = 无 */
    uint64_t engram_rows[DS4_V41_MAX_ENGRAM], engram_weight_off[DS4_V41_MAX_ENGRAM], engram_scale_off[DS4_V41_MAX_ENGRAM];
    char     engram_table_path[DS4_V41_MAX_ENGRAM][1024];
    uint32_t engram_max_ngram, engram_heads, engram_head_dim, engram_vocab, engram_cvocab, engram_pad;
    float    swiglu_limit;
    uint32_t mtp_towers, mtp_experts;   /* DSpark: 塔数 / 每塔专家数, 0 = 这份 GGUF 没带三塔 */
    uint32_t mtp_used;         /* 每个草稿位在每塔选几个专家(官方 3) */
    uint32_t mtp_block;        /* 一次出几个草稿位(官方 5); 0 = 元数据没带 ⇒ 投机路不武装 */
    uint32_t mtp_noise_id;     /* 草稿块首位之后那几位的占位 token id(官方 128799) */
    uint32_t mtp_markov_rank;  /* markov 头的秩(官方 256) */
    int16_t  mtp_target[DS4_MTP_MAX_TOWERS * 2];   /* main_x 取哪几层的注意力输入(官方 37/38/39) */
    int16_t  mtp_target_slot[DS4_MAX_LAYER];       /* 层 → 在 main_hidden 里的第几段, -1 = 不取 */
    uint32_t n_mtp_target;
} ds4_v41_cfg;
extern ds4_v41_cfg g_ds4_v41;

typedef struct {
    const char *name;
    ds4_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_out_group;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    float rms_eps;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float compress_rope_freq_base;
    uint64_t rope_orig_ctx;
} ds4_shape;

extern ds4_shape g_ds4_shape;

#define DS4_MODEL_SHAPE_NAME          (g_ds4_shape.name)
#define DS4_MODEL_VARIANT             (g_ds4_shape.variant)
#define DS4_N_LAYER                   (g_ds4_shape.n_layer)
#define DS4_N_EMBD                    (g_ds4_shape.n_embd)
#define DS4_N_VOCAB                   (g_ds4_shape.n_vocab)
#define DS4_N_HEAD                    (g_ds4_shape.n_head)
#define DS4_N_HEAD_KV                 (g_ds4_shape.n_head_kv)
#define DS4_N_HEAD_DIM                (g_ds4_shape.n_head_dim)
#define DS4_N_VALUE_DIM               (g_ds4_shape.n_value_dim)
#define DS4_N_ROT                     (g_ds4_shape.n_rot)
#define DS4_N_OUT_GROUP               (g_ds4_shape.n_out_group)
#define DS4_N_LORA_Q                  (g_ds4_shape.n_lora_q)
#define DS4_N_LORA_O                  (g_ds4_shape.n_lora_o)
#define DS4_N_EXPERT                  (g_ds4_shape.n_expert)
#define DS4_N_EXPERT_USED             (g_ds4_shape.n_expert_used)
#define DS4_N_EXPERT_SHARED           (g_ds4_shape.n_expert_shared)
#define DS4_N_FF_EXP                  (g_ds4_shape.n_ff_exp)
#define DS4_N_HASH_LAYER              (g_ds4_shape.n_hash_layer)
#define DS4_N_SWA                     (g_ds4_shape.n_swa)
#define DS4_N_INDEXER_HEAD            (g_ds4_shape.n_indexer_head)
#define DS4_N_INDEXER_HEAD_DIM        (g_ds4_shape.n_indexer_head_dim)
#define DS4_N_INDEXER_TOP_K           (g_ds4_shape.n_indexer_top_k)
#define DS4_N_HC                      (g_ds4_shape.n_hc)
#define DS4_N_HC_SINKHORN_ITER        (g_ds4_shape.n_hc_sinkhorn_iter)
#define DS4_RMS_EPS                   (g_ds4_shape.rms_eps)
#define DS4_HC_EPS                    (g_ds4_shape.hc_eps)
#define DS4_EXPERT_WEIGHT_SCALE       (g_ds4_shape.expert_weight_scale)
#define DS4_SWIGLU_CLAMP_EXP          (g_ds4_shape.swiglu_clamp_exp)
#define DS4_ROPE_FREQ_BASE            (g_ds4_shape.rope_freq_base)
#define DS4_ROPE_SCALE_FACTOR         (g_ds4_shape.rope_scale_factor)
#define DS4_ROPE_YARN_BETA_FAST       (g_ds4_shape.rope_yarn_beta_fast)
#define DS4_ROPE_YARN_BETA_SLOW       (g_ds4_shape.rope_yarn_beta_slow)
#define DS4_COMPRESS_ROPE_FREQ_BASE   (g_ds4_shape.compress_rope_freq_base)
#define DS4_ROPE_ORIG_CTX             (g_ds4_shape.rope_orig_ctx)

#define DS4_MAX_DIMS   8

typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef ds4_tokens token_vec;

enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I32      = 26,
    DS4_TENSOR_BF16     = 30,   /* 官方权重的原生精度: 转换器不再展开成 f32(clear.md C1, 2026-09-15) */
    DS4_TENSOR_GO1B     = 40,   /* strict-1-bit routed expert; mirrors GGUF ggml type 40 */
    DS4_TENSOR_GO2B     = 41,   /* merged base+residual binary pair (R5-C go2b) */
    DS4_TENSOR_VQBLOB   = 42,   /* VQ 层 blob(DQVL), 字节不透明 */
    DS4_TENSOR_FP4X32   = 43,
    DS4_TENSOR_FP8_32X32 = 44,  /* engram wkv: e4m3 + 32×32 块 ue8m0(src/common/ds4_quantfmt.h) */   /* V4.1 骨架: FP4 E2M1 + ue8m0/32, 17 B/32 元素(src/common/ds4_quantfmt.h) */
};

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;

    /* Reduced-expert ("keep-map") models: a shrunken GGUF stores only the kept
     * routed experts per layer (ffn_*_exps dim[2] = kept count, not DS4_N_EXPERT),
     * while the router still emits 256-wide logits. These map original expert id
     * -> compact slot in the shrunken tensor; -1 means the expert was dropped.
     * Populated by load_expert_keep_map(); empty/NULL for a full model. */
    bool      expert_shrunken;
    uint32_t  expert_layer_count;          /* layers covered by the keep-map */
    uint16_t *expert_kept_count;           /* [expert_layer_count]: kept experts per layer */
    int16_t  *expert_orig_to_compact;      /* [expert_layer_count * DS4_N_EXPERT]: orig id -> slot, or -1 */

    /* Optional go1b "hidden variable z^L" four-loss correction loaded from a
     * separate small sidecar GGUF (blk.{L}.corr_*). NULL => pure 1-bit (today's
     * behaviour). Owned by this model; freed in model_close(). */
    struct ds4_corr *corr;

    /* Optional go1b 1-bit RESIDUAL Q1(W-Q1(W)) sidecar (blk.{L}.ffn_*_exps_res).
     * NULL => single 1-bit. Owned by this model; freed in model_close(). */
    struct ds4_residual *residual;

    /* Optional go-onebit DQZ2 multiplicative correction chain (--zchain /
     * DS4_ZCHAIN): per-expert GE gains folded into the router weights + the
     * per-token routed scale λ(x). NULL => raw base (today's behaviour).
     * Owned by this model; freed in model_close(). */
    struct ds4_zchain *zchain;
} ds4_model;

/* Per-layer go1b correction tensors (host pointers into the sidecar mmap, plus
 * resident GPU copies for the Metal/CUDA forward).  Shapes (GGUF ne, inner dim
 * first): corr_U ne=[d_l,d_model], corr_V ne=[d_model,d_l], corr_C ne=[d_l,n_exp],
 * corr_b ne=[d_model], corr_beta/corr_delta ne=[n_exp].  d_l = corr_C ne[0]. */
typedef struct {
    bool present;
    bool has_delta;       /* any nonzero router-logit correction? δ≡0 sidecars
                             skip the per-layer corr_router_bias dispatch — on
                             the A3 offload decode path that dispatch is an
                             owned-CB commit+wait per layer (pure sync cost). */
    uint32_t d_l;
    const float *U;       /* [d_model][d_l] row-major */
    const float *V;       /* [d_l][d_model] row-major */
    const float *C;       /* [n_exp][d_l]   row-major */
    const float *b;       /* [d_model] */
    const float *beta;    /* [n_exp] */
    const float *delta;   /* [n_exp] */
#ifndef DS4_NO_GPU
    ds4_gpu_tensor *gU, *gV, *gC, *gb, *gbeta, *gdelta;   /* resident GPU copies */
#endif
} ds4_corr_layer;

struct ds4_corr {
    bool present;             /* at least one layer carries a correction */
    bool phi_yhat;            /* latent features φ: false = ffn_norm (x, legacy),
                                 true = routed_out (ŷ) — the solver's --feat yhat.
                                 Same kernel either way (phase1 reads the whole φ row
                                 into threadgroup memory before the barrier, phase2
                                 writes out after it, so out-aliasing is safe). */
    ds4_model sidecar;        /* the opened sidecar GGUF, kept mmap'd for host pointers */
    ds4_corr_layer layer[DS4_MAX_LAYER];
};

/* Per-layer go1b 1-bit residual expert tensors (sidecar mmap views, go1b type 40).
 * The routed-MoE forward runs a SECOND mm_id over these and sums into the base
 * expert output: W ≈ dequant(base go1b) + dequant(residual go1b). base_cos 0.52→0.82. */
typedef struct {
    bool present;
    ds4_tensor *gate, *up, *down;   /* blk.{L}.ffn_{gate,up,down}_exps_res.weight (sidecar) */
    ds4_tensor *lut;                /* blk.{L}.ffn_res_lut.weight (F32[256]); NULL if dense */
    const void *vq_raw; size_t vq_sz;   /* v2.2 直读侧车模式: dql_vq_L%02d.bin mmap(免 overlay 复制) */
#ifndef DS4_NO_GPU
    ds4_gpu_tensor *g_gate, *g_up, *g_down;   /* RESIDENT GPU copies (offload mmap reads 0) */
#endif
} ds4_residual_layer;

struct ds4_residual {
    bool present;
    ds4_model sidecar;
    ds4_residual_layer layer[DS4_MAX_LAYER];
};


/* ---- ds4.c internals published to modules (definitions stay in ds4.c) ---- */
void ds4_die(const char *msg);
void *xcalloc(size_t n, size_t size);
void *xmalloc(size_t size);
void model_open(ds4_model *m, const char *path, bool metal_mapping, bool prefetch_cpu);
void model_close(ds4_model *m);
bool model_get_bool(const ds4_model *m, const char *key, bool *out);
ds4_tensor *model_find_tensor(const ds4_model *m, const char *name);
const void *tensor_data(const ds4_model *m, const ds4_tensor *t);

/* ---- ds4_corr.c (go1b z-sidecar module) ---- */
void corr_free(struct ds4_corr *corr);
struct ds4_corr *corr_load(const char *path, bool metal_mapping);
const float *corr_layer_delta(const ds4_model *m, uint32_t il);
void corr_apply_moe_host(float *out, const ds4_model *m, uint32_t il,
                         const float *x, const int *selected, uint32_t n_sel);

#endif /* DS4_INTERNAL_H */
