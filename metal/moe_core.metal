// DS4 Metal routed-MoE matvec kernels.

#define QK_K 256
#define N_R0_Q2_K 4
#define N_R0_Q4_K 2
#define N_R0_IQ2_XXS 4

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

// GO1B: strict-binary 1-bit routed-expert weight (DS4 go-onebit). 256-element block =
// one fp16 row scale (replicated into every block of the row) + 256 sign bits.
// element j sign at signs[j/8] bit (j%8): bit==1 -> +d (encoder packs w>=0 as 1), bit==0 -> -d.
struct block_go1b {
    half d;
    uint8_t signs[QK_K/8];   // 32 bytes = 256 sign bits
};

// GO2B: offline-merged base+residual strict-binary pair (R5-C). 256-element block =
// base row scale d1 + residual per-block scale d2 + both sign planes. Element value
// is the EXACT sum (±d1)+(±d2) of the two go1b layers it replaces, so a single
// mm_id pass over go2b hot experts is bit-equivalent to the two-pass add path.
struct block_go2b {
    half d1;
    half d2;
    uint8_t s1[QK_K/8];      // base sign bits
    uint8_t s2[QK_K/8];      // residual sign bits
};                            // 68 bytes / 256 weights = 2.125 bpw

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

struct ds4_metal_dsv4_moe_sum6_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

// Routed-MoE activation for the selected experts:
// clamp(gate), clamp(up), silu(gate) * up * route_weight.  Normal inference
// does not consume gate/up after this point, so the fast path avoids writing the
// clamped intermediates back.  A diagnostic env switch can restore those writes
// when comparing the old multi-kernel intermediate tensors.
kernel void kernel_dsv4_moe_swiglu_weight(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device float *mid_row  = (device float *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = silu * u * route_weight;
    }
}

// Same routed-MoE activation as above, but stores the down-projection input in
// half precision. The grouped matmul path converts F32 activations to half
// before MMA anyway, so this cuts the large mid write/read traffic without
// changing the effective matmul input precision.
kernel void kernel_dsv4_moe_swiglu_weight_f16(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device half  *mid_row  = (device half  *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = (half)(silu * u * route_weight);
    }
}

kernel void kernel_dsv4_moe_sum6_f32(
        constant ds4_metal_dsv4_moe_sum6_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float v = s[col];
        v += s[args.width + col];
        v += s[2u * args.width + col];
        v += s[3u * args.width + col];
        v += s[4u * args.width + col];
        v += s[5u * args.width + col];
        d[col] = v;
    }
}

/* REAP (Router-weighted Expert Activation Pruning, arXiv:2510.13999) saliency.
 * The sum6 input holds each selected expert's GATED output (route_weight is folded
 * in at swiglu_weight, before the down-projection), so ||slot_k||_2 == g_k*||f_k(x)||_2
 * -- exactly REAP's per-token contribution g_j(x)*||f_j(x)||_2.  This kernel writes
 * the per-(token,slot) L2 norm; the host then accumulates the REAP saliency
 * S_j = (1/|X_j|) sum_{x in X_j} g_j(x)*||f_j(x)||_2, keyed by the routed expert id. */
struct ds4_metal_dsv4_reap_args {
    uint32_t tokens;
    uint32_t width;            // output dim per expert (sum6 column count)
    uint32_t n_used;           // selected experts per token (6)
    uint32_t layer;            // layer index, for the [layer*256 + expert] accumulator slot
    uint64_t src_token_stride; // bytes per token in src (== n_used*width*4)
};

/* Accumulate REAP saliency on-GPU: for each (token, slot) compute ||gated output||_2
 * (== g_j*||f_j(x)||_2) and atomic-add it (fixed-point x256) into saliency[layer*256+e]
 * keyed by the routed expert id, plus a token count.  One readback at exit yields
 * S_j = saliency / (256 * count).  Fixed-point avoids atomic_float portability issues;
 * the magnitudes here stay well within uint32 for a domain calibration set. */
kernel void kernel_dsv4_reap_accum(
        constant ds4_metal_dsv4_reap_args &args,
        device const char  *src,        // sum6 input: per token, n_used gated expert outputs
        device const int   *active_ids, // [tokens*n_used] slot -> expert id (0..255, -1 = none)
        device atomic_uint *saliency,   // [n_layers*256] fixed-point accumulator
        device atomic_uint *counts,     // [n_layers*256] active-token counts
        uint gid[[thread_position_in_grid]]) {
    const uint total = args.tokens * args.n_used;
    if (gid >= total) return;
    const uint token = gid / args.n_used;
    const uint slot  = gid % args.n_used;
    device const float *s = (device const float *)(src + (uint64_t)token * args.src_token_stride)
                            + (uint64_t)slot * (uint64_t)args.width;
    float sumsq = 0.0f;
    for (uint c = 0; c < args.width; c++) { float v = s[c]; sumsq += v * v; }
    const int e = active_ids[gid];
    if (e >= 0 && e < 256) {
        const uint base = args.layer * 256u + (uint)e;
        atomic_fetch_add_explicit(&saliency[base], (uint)(sqrt(sumsq) * 256.0f), memory_order_relaxed);
        atomic_fetch_add_explicit(&counts[base], 1u, memory_order_relaxed);
    }
}

