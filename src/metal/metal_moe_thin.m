/* metal_moe_thin.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* Minimum batch token count for the grouped-GEMM (mm_id) routed-MoE path.  Was
 * an in-function static in ds4_gpu_routed_moe_batch_tensor; extracted so the
 * P-OVL two-pass path can gate on the same threshold (overlap only applies to
 * the mm_id encode shape). */
uint32_t ds4_gpu_moe_mm_id_min(void) {
    static uint32_t cached;
    static int init;
    if (!init) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_MOE_MM_ID_MIN", 8u);
        if (v == 0) v = UINT32_MAX;     /* 0 = never use mm_id */
        if (v < 2u) v = 2u;
        cached = (uint32_t)(v > UINT32_MAX ? UINT32_MAX : v);
        init = 1;
    }
    return cached;
}

/* project.md §2.4/§3.5 降激活 (MoE activation thinning) -- the one lever that moves
 * the dominant q2 wall (routed-expert SSD IO ~1.70GiB/token).  After routing has
 * already chosen the per-token top-n_expert experts (selectedbuf ids + weightsbuf
 * route weights), optionally drop the low-weight tail picks so the gather *union*
 * shrinks and the per-token cold-expert bytes fall ~proportionally.  Cutting k 6->K
 * lifts single-forward decode toward the W3 backbone ceiling (~10-18 t/s) and prefill
 * toward the M2/M3 targets; combined with §3.5 copy-spec it is the only code path to
 * the 30 t/s effective target (project.md §1.5 / wave 66).
 *
 * Mechanism (no kernel change, IO actually saved):
 *   a dropped pick is rewritten to ALIAS the token's top-1 expert id with route
 *   weight 0, so (1) the grouped GEMM computes top1_swiglu*0 = 0 -- the pick
 *   contributes nothing -- and (2) ds4_gpu_collect/compact_selected_experts dedups
 *   it away, so the dropped expert is never gathered.  Kept weights are rescaled to
 *   the original sum of the n_expert routed weights (magnitude-neutral redistribution
 *   of the dropped mass).
 *
 * This changes model outputs => a quality bet.  Two independent gates, BOTH default
 * OFF => byte-identical baseline (kept off, the rewrite never runs):
 *   DS4_METAL_MOE_THIN_ALPHA  (float)  drop picks with weight < alpha * top1_weight
 *   DS4_METAL_MOE_THIN_TOPK   (uint)   keep at most this many highest-weight picks
 * Validated by the user via ds4-eval q1..q4 --temp 0 --seed 1 + ds4_test
 * --logprob-vectors before it can ever default on. */
static float    g_moe_thin_alpha = -1.0f;

/* <0 = env unread */
static uint32_t g_moe_thin_topk;

/* Wave 68: only thin batches with >= this many tokens.  Routed-MoE batches come
 * in two shapes: large prefill chunks (hundreds-thousands of tokens) and small
 * copy-spec verify batches (<= COPY_SPEC_DRAFT, 64) / single-token decode (1).
 * Thinning the verify/decode path changes the greedy argmax and collapses
 * copy-spec verbatim acceptance (wave 67: code-edit tok/call 5.39->1.1, decode
 * -27%).  Setting MIN_TOKENS above the verify cap (e.g. 128) makes thinning
 * prefill-only: it keeps the +30% prefill IO win while leaving copy-spec decode
 * byte-identical.  Default 1 = thin every batch (preserves the wave-66 knob). */
static uint32_t g_moe_thin_min_tokens = 1u;

/* Wave 69: upper gate, complement of min_tokens.  0 = no cap (legacy).  Set to 1
 * to thin ONLY single-token decode (n_tokens==1).  In copy-spec mode that is
 * exactly the no-match bare round: matched positions ride a verify batch
 * (n_tokens = 1+n_copy >= 2) which stays full top-k / byte-identical, so
 * copy-spec verbatim acceptance is preserved (the wave-67 collapse came from
 * thinning the verify batch; this avoids it).  Run-A measured no-match decode is
 * SSD-bound on cold expert gathers (pread 96-140ms/cold-layer); halving picks
 * there cuts that IO where copy-spec cannot help.  Quality bet on the novel
 * tokens only.  Default 0 => unchanged. */
static uint32_t g_moe_thin_max_tokens = 0u;

static void ds4_gpu_moe_thin_init(void) {
    if (g_moe_thin_alpha >= 0.0f) return;
    const char *a = getenv("DS4_METAL_MOE_THIN_ALPHA");
    const char *k = getenv("DS4_METAL_MOE_THIN_TOPK");
    const char *m = getenv("DS4_METAL_MOE_THIN_MIN_TOKENS");
    const char *x = getenv("DS4_METAL_MOE_THIN_MAX_TOKENS");
    float av = (a && *a) ? (float)atof(a) : 0.0f;
    g_moe_thin_alpha = av > 0.0f ? av : 0.0f;
    g_moe_thin_topk  = (k && *k) ? (uint32_t)strtoul(k, NULL, 10) : 0u;
    g_moe_thin_min_tokens = (m && *m) ? (uint32_t)strtoul(m, NULL, 10) : 1u;
    if (g_moe_thin_min_tokens < 1u) g_moe_thin_min_tokens = 1u;
    g_moe_thin_max_tokens = (x && *x) ? (uint32_t)strtoul(x, NULL, 10) : 0u;
    if (g_moe_thin_alpha > 0.0f || g_moe_thin_topk > 0u) {
        fprintf(stderr,
                "ds4: MoE activation thinning enabled (alpha=%.4f topk=%u min_tokens=%u max_tokens=%u) -- quality bet, "
                "gate with q1..q4 + --logprob-vectors\n",
                g_moe_thin_alpha, g_moe_thin_topk, g_moe_thin_min_tokens, g_moe_thin_max_tokens);
    }
}

static inline int ds4_gpu_moe_thin_enabled(void) {
    ds4_gpu_moe_thin_init();
    return (g_moe_thin_alpha > 0.0f || g_moe_thin_topk > 0u) ? 1 : 0;
}

/* Rewrite selectedbuf/weightsbuf in place on the CPU.  Must be called after the
 * routing kernels are drained (selectedbuf CPU-valid) and before the gather/union
 * and before the swiglu/mm_id GEMM is encoded (they re-read both buffers).  No-op
 * unless a thinning gate is set, or unless either buffer is not Shared. n_expert<=6. */
void ds4_gpu_moe_thin_picks(id<MTLBuffer> selectedbuf, NSUInteger selected_off,
                                   id<MTLBuffer> weightsbuf, NSUInteger weights_off,
                                   uint32_t n_tokens, uint32_t n_expert,
                                   uint32_t gate_type) {
    if (!ds4_gpu_moe_thin_enabled()) return;
    /* Thinning was calibrated on q2 (robust experts, +2.7% code-edit). On
     * strict 1/2-bit experts (go1b/go2b) dropping routed picks is the
     * adjudicated-dead "expert-count pruning": measured 2026-07-06, topk=4 on
     * the mono model turns byte-correct greedy Go into word soup. Refuse it
     * at the engine level so no launcher default can re-break quality. */
    if (gate_type == DS4_METAL_TENSOR_GO1B || gate_type == DS4_METAL_TENSOR_GO2B) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                    "ds4: MoE thinning requested but routed experts are strict 1/2-bit "
                    "(go1b/go2b) -- refusing (quality-catastrophic, measured); full top-k kept\n");
        }
        return;
    }
    if (n_expert <= 1u || n_tokens == 0u) return;
    if (n_tokens < g_moe_thin_min_tokens) return;   /* prefill-only gate (wave 68) */
    if (g_moe_thin_max_tokens != 0u && n_tokens > g_moe_thin_max_tokens) return; /* no-match-decode-only gate (wave 69) */
    if (!selectedbuf || !weightsbuf) return;
    if (selectedbuf.storageMode != MTLStorageModeShared ||
        weightsbuf.storageMode != MTLStorageModeShared) return;
    int32_t *sel = (int32_t *)((uint8_t *)selectedbuf.contents + (size_t)selected_off);
    float   *wt  = (float   *)((uint8_t *)weightsbuf.contents  + (size_t)weights_off);
    const float    alpha = g_moe_thin_alpha;
    const uint32_t topk  = g_moe_thin_topk;
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t *s = sel + (size_t)t * n_expert;
        float   *w = wt  + (size_t)t * n_expert;
        uint32_t top1 = 0;
        float    top1w = w[0];
        float    sum_all = w[0];
        for (uint32_t i = 1; i < n_expert; i++) {
            sum_all += w[i];
            if (w[i] > top1w) { top1w = w[i]; top1 = i; }
        }
        if (!(top1w > 0.0f) || !(sum_all > 0.0f)) continue;  /* degenerate: leave token */
        const int32_t top1_id = s[top1];
        const float   thresh  = alpha * top1w;
        float    sum_kept = 0.0f;
        uint32_t kept     = 0;
        for (uint32_t i = 0; i < n_expert; i++) {
            int drop = 0;
            if (i != top1) {
                if (alpha > 0.0f && w[i] < thresh) drop = 1;
                if (topk > 0u) {
                    uint32_t greater = 0;   /* picks strictly ahead of i (ties -> lower index wins) */
                    for (uint32_t j = 0; j < n_expert; j++)
                        if (w[j] > w[i] || (w[j] == w[i] && j < i)) greater++;
                    if (greater >= topk) drop = 1;
                }
            }
            if (drop) {
                s[i] = top1_id;   /* alias an expert the union already gathers */
                w[i] = 0.0f;
            } else {
                sum_kept += w[i];
                kept++;
            }
        }
        if (kept < n_expert && sum_kept > 0.0f) {
            const float scale = sum_all / sum_kept;   /* preserve original routed mass */
            for (uint32_t i = 0; i < n_expert; i++)
                if (w[i] != 0.0f) w[i] *= scale;
        }
    }
}

/* Coverage-cliff simulation (DS4_EXPERT_KEEP_SIM=<file>): drop router picks
 * whose expert is outside the per-layer kept set (43 text lines of kept ids),
 * exactly mirroring the keep-map product semantics (alias to top1, zero
 * weight, renormalize kept mass). Bit-width untouched — isolates the coverage
 * variable for the 2bit-keepmap go/no-go verdict. Default off. */
static uint8_t g_keep_sim[64][256];

static int g_keep_sim_on = -1;

void ds4_gpu_moe_sim_keepmap(id<MTLBuffer> selectedbuf, NSUInteger selected_off,
                                    id<MTLBuffer> weightsbuf, NSUInteger weights_off,
                                    uint32_t n_tokens, uint32_t n_expert, uint32_t layer) {
    if (g_keep_sim_on < 0) {
        g_keep_sim_on = 0;
        const char *p = getenv("DS4_EXPERT_KEEP_SIM");
        if (p && p[0]) {
            FILE *f = fopen(p, "r");
            if (f) {
                char line[4096];
                uint32_t L = 0;
                while (L < 64 && fgets(line, sizeof(line), f)) {
                    char *tok = strtok(line, " \t\n");
                    while (tok) {
                        int e = atoi(tok);
                        if (e >= 0 && e < 256) g_keep_sim[L][e] = 1;
                        tok = strtok(NULL, " \t\n");
                    }
                    L++;
                }
                fclose(f);
                g_keep_sim_on = 1;
                fprintf(stderr, "ds4: expert keep-sim active (%u layers from %s)\n", L, p);
            }
        }
    }
    if (g_keep_sim_on != 1 || layer >= 64 || n_expert <= 1u || n_tokens == 0u) return;
    if (!selectedbuf || !weightsbuf) return;
    if (selectedbuf.storageMode != MTLStorageModeShared ||
        weightsbuf.storageMode != MTLStorageModeShared) return;
    int32_t *sel = (int32_t *)((uint8_t *)selectedbuf.contents + (size_t)selected_off);
    float   *wt  = (float   *)((uint8_t *)weightsbuf.contents  + (size_t)weights_off);
    const uint8_t *keep = g_keep_sim[layer];
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t *s = sel + (size_t)t * n_expert;
        float   *w = wt  + (size_t)t * n_expert;
        float sum_all = 0.0f, sum_kept = 0.0f;
        int32_t anchor_id = -1;
        float   anchor_w  = -1.0f;
        for (uint32_t i = 0; i < n_expert; i++) {
            sum_all += w[i];
            if (s[i] >= 0 && s[i] < 256 && keep[s[i]] && w[i] > anchor_w) {
                anchor_w = w[i]; anchor_id = s[i];
            }
        }
        if (anchor_id < 0) continue;   /* every pick outside kept set: leave token (counted rare) */
        for (uint32_t i = 0; i < n_expert; i++) {
            if (s[i] >= 0 && s[i] < 256 && !keep[s[i]]) {
                s[i] = anchor_id;      /* alias to a kept expert the union gathers */
                w[i] = 0.0f;
            } else {
                sum_kept += w[i];
            }
        }
        if (sum_kept > 0.0f && sum_kept < sum_all) {
            const float scale = sum_all / sum_kept;
            for (uint32_t i = 0; i < n_expert; i++)
                if (w[i] != 0.0f) w[i] *= scale;
        }
    }
}

/* P-OVL (expert-half overlap, bit-exact): split a verify batch's active expert
 * set into two disjoint scratch slot-ranges, commit pass 0 (GPU starts), then
 * gather pass 1 on the CPU while the GPU runs pass 0 -- the routed-expert disk
 * gather (the decode bottleneck) overlaps the GEMM.  Default OFF: it only
 * restructures the mm_id verify path, leaving the 3.77 baseline byte-identical
 * until DS4_METAL_MOE_OVERLAP=1. */
int ds4_gpu_moe_overlap_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = ds4_gpu_env_bool("DS4_METAL_MOE_OVERLAP") > 0 ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                    "ds4: routed-MoE expert-half overlap enabled "
                    "(DS4_METAL_MOE_OVERLAP=1, bit-exact two-pass)\n");
        }
    }
    return cached;
}

/* Below this active-expert count a split is not worth a second CB commit
 * (the gather is tiny and the flush overhead dominates); fall back to the
 * single-pass path.  A/B via DS4_METAL_MOE_OVERLAP_MIN_EXPERTS. */
uint32_t ds4_gpu_moe_overlap_min_experts(void) {
    static uint32_t cached;
    static int init;
    if (!init) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_MOE_OVERLAP_MIN_EXPERTS", 8u);
        if (v < 2u) v = 2u;
        if (v > 1024u) v = 1024u;
        cached = (uint32_t)v;
        init = 1;
    }
    return cached;
}

/* Number of overlap passes (slot-range chunks).  Total decode forward time ~=
 * T_gather + T_gpu/N: each extra pass hides another 1/N of the GEMM behind the
 * disk gather (the physical floor), at the cost of one more CB commit + one
 * more full-pair_rows swiglu + an N-times-larger empty-expert dispatch grid.
 * Default 2 = the wave-41 validated behavior; A/B 3/4 to chase the knee. */
uint32_t ds4_gpu_moe_overlap_passes(void) {
    static uint32_t cached;
    static int init;
    if (!init) {
        uint64_t v = ds4_gpu_env_u64("DS4_METAL_MOE_OVERLAP_PASSES", 2u);
        if (v < 1u) v = 1u;
        if (v > 8u) v = 8u;
        cached = (uint32_t)v;
        init = 1;
    }
    return cached;
}
