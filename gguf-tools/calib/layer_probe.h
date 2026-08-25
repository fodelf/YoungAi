/* layer_probe.h — per-expert SwiGLU FFN forward + 1-bit reconstruction probe.
 *
 * Part of the Go-domain 1-bit quantizer (gguf-tools/go-onebit/, SPEC.md §5/§6).
 * Computes, for each routed expert e, the full-precision output o_e and the
 * 1-bit output ô_e on a pool of Go activations {x_i}, plus the residual
 * Δo_e = o_e − ô_e. Stacking {Δo_{e,i}} per layer feeds the power-iteration
 * rank estimate (SPEC §6.3 → table P).
 *
 * DeepSeek-V4-Flash expert geometry (config.json): d_model(hidden)=4096,
 * d_ff(moe_intermediate)=2048, SwiGLU MoE expert:
 *     o = down · ( SiLU(gate·x) ⊙ (up·x) ),  SiLU(z)=z·sigmoid(z).
 * Row-major tensors:  gate,up : [d_ff × d_model]   down : [d_model × d_ff].
 *
 * This module is DECOUPLED from the 1-bit codec (onebit_quant.c, built in
 * parallel): it never references go1b_* symbols. The packed-weight forward
 * reaches the codec only through a caller-supplied dequant_row_fn callback,
 * which is signature-compatible with go1b_dequantize_row().
 *
 * Pure C99. No C++.
 */
#ifndef GO_ONEBIT_LAYER_PROBE_H
#define GO_ONEBIT_LAYER_PROBE_H

#include <stddef.h>   /* size_t  */
#include <stdint.h>   /* int64_t */

/* Full-precision reference expert forward (this output is "o_ref").
 *   x    : [d_model]
 *   gate : [d_ff   × d_model]  row-major
 *   up   : [d_ff   × d_model]  row-major
 *   down : [d_model × d_ff]    row-major
 *   o    : [d_model]           caller-allocated output
 * See layer_probe.c for the SwiGLU shape-flow commentary. */
void expert_forward_f32(const float *x,
                        const float *gate, const float *up, const float *down,
                        float *o, int d_model, int d_ff);

/* Same, with the runtime's SwiGLU activation clamp (g=min(g,lim),
 * u=clamp(u,±lim); lim<=0 → no clamp). Matches moe.metal
 * kernel_dsv4_moe_swiglu_weight and the teacher capture's SWLIM. */
void expert_forward_f32_lim(const float *x,
                            const float *gate, const float *up, const float *down,
                            float *o, int d_model, int d_ff, float lim);

/* On-the-fly row dequant callback. Signature-identical to the codec's
 * go1b_dequantize_row(): decode one packed weight row into `dst` (ncols
 * floats). The probe stays decoupled — pass &go1b_dequantize_row (or any
 * compatible decoder) at the call site; this header links nothing. */
typedef void (*dequant_row_fn)(const void *row, float *dst, int64_t ncols);

/* 1-bit-consuming expert forward (this output is "ô"). Identical SwiGLU math
 * to expert_forward_f32, but each weight row is decoded on the fly via `deq`.
 * The *_q pointers are opaque bases of row-record arrays; row r of a matrix is
 * at base + (size_t)r * row_bytes_*, so the caller's packed-record stride
 * (e.g. go1b_row_bytes(ncols)) is honored without this module knowing the
 * codec layout.
 *   gate_q,up_q : d_ff   rows, each decodes to d_model floats
 *   down_q      : d_model rows, each decodes to d_ff   floats */
void expert_forward_dequant(const float *x,
                            const void *gate_q, const void *up_q, const void *down_q,
                            dequant_row_fn deq,
                            size_t row_bytes_gate, size_t row_bytes_up, size_t row_bytes_down,
                            float *o, int d_model, int d_ff);

/* One expert's weights as seen by the probe. The full-precision pointers feed
 * expert_forward_f32 (o_ref); the packed *_q pointers + row strides feed
 * expert_forward_dequant (ô). A loader may supply either or both forms; NULL
 * pointers disable the corresponding output. */
typedef struct {
    const float *gate;            /* [d_ff   × d_model] f32 row-major, or NULL */
    const float *up;              /* [d_ff   × d_model] f32 row-major, or NULL */
    const float *down;            /* [d_model × d_ff]   f32 row-major, or NULL */
    const void  *gate_q;          /* packed 1-bit row records,         or NULL */
    const void  *up_q;            /* packed 1-bit row records,         or NULL */
    const void  *down_q;          /* packed 1-bit row records,         or NULL */
    size_t       row_bytes_gate;  /* stride between packed gate rows */
    size_t       row_bytes_up;    /* stride between packed up   rows */
    size_t       row_bytes_down;  /* stride between packed down rows */
} expert_weights;

/* Typed loader hook. Fill `out` with expert `expert`'s weights at transformer
 * layer `layer`. Return 0 on success, non-zero to skip the expert. `user`
 * carries loader context (e.g. an mmap'd GGUF handle). MUST be thread-safe:
 * probe_experts may call it concurrently from worker threads. The real GGUF/HF
 * binding is out of scope here — see the TODO(integration) in layer_probe.c. */
typedef int (*expert_loader_fn)(void *user, int layer, int expert, expert_weights *out);

/* Probe every expert of one layer over an activation pool.
 *
 *   x_pool       : [n_x × d_model] row-major Go activations
 *   load/load_user : per-expert weight access hook (see expert_loader_fn)
 *   deq          : packed-row decoder used for ô (NULL → skip ô / residual)
 *   n_threads    : parallelism over experts (<=0 → 1; clamped to n_experts),
 *                  mirroring the engine's expert_worker pattern
 *
 * Outputs (each NULL to skip), laid out as [n_experts][n_x][d_model]; element
 * (e,i) starts at ((size_t)e*n_x + i)*d_model:
 *   oref_out     : full-precision o_ref   (needs f32 weights)
 *   ohat_out     : 1-bit ô                (needs packed weights + deq)
 *   residual_out : Δo = o_ref − ô         (needs both of the above)
 */
void probe_experts(const float *x_pool, int n_x,
                   int layer, int n_experts,
                   expert_loader_fn load, void *load_user,
                   dequant_row_fn deq,
                   int d_model, int d_ff,
                   int n_threads,
                   float *oref_out,
                   float *ohat_out,
                   float *residual_out);

#endif /* GO_ONEBIT_LAYER_PROBE_H */
