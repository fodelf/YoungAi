/* =========================================================================
 * ds4_zchain -- go-onebit DQZ2 runtime sidecar: the per-layer multiplicative
 * correction chain landed by the quantizer, replayed at inference.
 * =========================================================================
 *
 * The quantizer (gguf-tools/go-onebit/quant/ds4quant_run.c) tunes each layer's
 * routed-MoE output with a chain of closed-form multiplicative ops and lands
 * them as vd=1 records in the per-layer DQL2 files; zchain_write() serializes
 * the FINAL chains (post backward-fixing) into one small "DQZ2" file. This
 * module loads that file and reduces each layer's chain to two runtime hooks,
 * byte-faithful to the quantizer's bytes_moe() replay:
 *
 *   GE  (type 5): per-expert gain, multiplied into the router gate weight of
 *                 the selected expert BEFORE the expert matmuls (the quantizer
 *                 applies the LAST type-5 record only; so do we).
 *   λ(x) (types 1..4): every scaling op re-anchors on the shared-expert base
 *                 and TREF re-anchors on the post-expert state, so the whole
 *                 chain algebraically collapses to ONE per-token scalar on the
 *                 routed contribution:
 *                     F = shared_out + λ(x) * Σ ge-weighted expert outputs
 *                 folded sequentially over the record order:
 *                     1 GL     λ <- g * λ
 *                     2 GLdyn2 λ <- c * λ,  c = clamp(w0 + w1*(|x|2 - μ)/σ, 0.25, 4)
 *                     3 GLdyn8 λ <- c * λ,  c = clamp(w8[0] + Σk w8[1+k]*(V8_k · x), 0.25, 4)
 *                              (records without the V8 payload are skipped -- quantizer parity)
 *                     4 TREF   λ <- 1 + t * (λ - 1)
 *                 x = the token's MoE input (post-ffn_norm activation), the
 *                 same vector the router consumed.
 *
 * Pure host C: no GPU coupling. The Metal path uploads the same tables and
 * evaluates λ in-kernel (ds4_gpu_zchain_*); the CPU path calls
 * ds4_zchain_lambda() directly. Absent file / no ops => exactly today's path. */
#ifndef DS4_ZCHAIN_H
#define DS4_ZCHAIN_H

#include <stdint.h>
#include <stddef.h>
#include "ds4_z.h"   /* z 隐变量模块: 反修解算与引擎 apply 同一份实现(2026-08-26) */

typedef struct {
    uint32_t        type;   /* 1 GL | 2 GLdyn2 | 3 GLdyn8 | 4 TREF (GE is pre-resolved, not chained) */
    float           g;      /* type 1: gain; type 4: t */
    float           w2p[4]; /* type 2: w0, w1, feature mean, feature sd */
    float           w8[9];  /* type 3: intercept + 8 weights */
    const uint16_t *v8;     /* type 3: fp16 [8][d_model] projection rows (points into the mmap), or NULL */
} ds4_zchain_op;

/* Frozen z^L (type 6, 2026-07-14, 产物③): rank-k additive direction fix on the
 * routed sum, applied AFTER the λ scale (record order parity with bytes_moe):
 *     routed += clip * U diag(z) V^T x,  clip = min(1, tr*|routed|2 / |Δ|2)
 * Payload (fp16, aliases the sidecar mmap / model tensor): z[k] | U[d*k] | V[d*k],
 * U/V row-major [dim][k]. k <= 16. */
typedef struct {
    uint32_t        zlk;    /* active rank k (0 = absent) */
    uint32_t        zdin;   /* V input dim: d_model=linear | 3*d_model=ftA feature lift (md86) */
    float           zltr;   /* trust-region cap factor */
    uint32_t        zmul;   /* 0=加性 z^L(type6) | 2=rte 路由形态标记; type7/9 已删(2026-08-26 清仓) */
    const uint16_t *zlm;    /* fp16 z[k] | U[d_model*k] | V[zdin*k] */
    ds4_z          *zmod;   /* din==d_model 的线性 z: 载入时转 f32, apply 走 ds4_z 模块
                             * (与反修解算器同一份实现); din=3d ftA 走下方 fp16 旧路 */
} ds4_zchain_zl;

typedef struct {
    ds4_zchain_op *ops;     /* chain in record order (types 1..4 only) */
    uint32_t       n_ops;
    float         *ge;      /* [n_expert] effective per-expert gains (last type-5 record), or NULL */
    ds4_zchain_zl  zl;      /* frozen z^L (zlk==0 when absent) */
    ds4_zchain_zl  rte;     /* 路由闭式侧车(type8 zl.RTE, 2026-08-19): 结构同构复用,
                             * 语义 δlogits = U·tanh(Vᵀx/s) 加在 router raw logits 上
                             * (select 前)。zdin=d_model, U 行数=n_expert(非 d_model)。 */
} ds4_zchain_layer;

typedef struct ds4_zchain {
    uint32_t          n_layer;
    uint32_t          n_expert;
    uint32_t          d_model;
    ds4_zchain_layer *layer;      /* [n_layer] */
    void             *map;        /* whole-file mmap; v8 pointers alias into it */
    size_t            map_size;
    uint32_t          n_ops_total;
    uint32_t          n_ge_layers;
} ds4_zchain;

/* Load a DQZ2 sidecar. Returns NULL (and logs to stderr) on any mismatch --
 * missing file, bad magic, or a layer count different from n_layer. */
ds4_zchain *ds4_zchain_load(const char *path, uint32_t n_layer, uint32_t n_expert, uint32_t d_model);
void ds4_zchain_free(ds4_zchain *z);

/* Fold the layer's chain into the per-token routed scale. x is the token's
 * d_model MoE input (post-ffn_norm). Returns 1.0f when the layer has no ops. */
float ds4_zchain_lambda(const ds4_zchain *z, uint32_t il, const float *x);

static inline const float *ds4_zchain_layer_ge(const ds4_zchain *z, uint32_t il) {
    return (z && il < z->n_layer) ? z->layer[il].ge : (const float *)0;
}
static inline int ds4_zchain_layer_has_lambda(const ds4_zchain *z, uint32_t il) {
    return z && il < z->n_layer && z->layer[il].n_ops > 0;
}
static inline const ds4_zchain_zl *ds4_zchain_layer_zl(const ds4_zchain *z, uint32_t il) {
    return (z && il < z->n_layer && z->layer[il].zl.zlk) ? &z->layer[il].zl : (const ds4_zchain_zl *)0;
}
static inline const ds4_zchain_zl *ds4_zchain_layer_rte(const ds4_zchain *z, uint32_t il) {
    return (z && il < z->n_layer && z->layer[il].rte.zlk) ? &z->layer[il].rte : (const ds4_zchain_zl *)0;
}

/* Host-side frozen z^L apply: routed += clip * U diag(z) V^T x (contract above).
 * routed = the token's post-λ routed sum, modified in place. */
void ds4_zchain_zl_apply(const ds4_zchain_zl *zl, uint32_t d_model, const float *x, float *routed);

#endif /* DS4_ZCHAIN_H */
