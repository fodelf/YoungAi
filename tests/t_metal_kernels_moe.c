/* t_metal_kernels_moe.c — Metal kernel 数值回归中半: Q8_0 rowslice TP + go1b/go2b routed-MoE (机械拆分自 tests/ds4_test.c, 重构阶段8)。 */
#include "test_internal.h"
#ifndef DS4_NO_GPU

/* TP Phase 1 correctness gate: the row-parallel Q8_0 matvec split must reconstruct
 * the full matvec after the (host) all-reduce. Splits the in-dim into two halves,
 * computes each peer's partial out via ds4_gpu_matmul_q8_0_rowslice_tensor (the new
 * primitive: ne00=half blocks accumulated, nb01=full row stride for addressing),
 * sums them, and checks against the CPU reference. The two-half sum differs from a
 * single-machine reduction only in float associativity (~1e-6) — well inside the
 * Q8_0 tolerance. Proves the shared-FFN TP down projection before any dual-host run. */
void test_metal_q8_0_rowslice_tp(void) {
    const uint32_t in_dim = 128;          /* %64==0 so each half stays 32-block aligned */
    const uint32_t out_dim = 64;
    const uint32_t half = in_dim / 2u;    /* 64; 2 Q8_0 blocks per half-row */
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t half_bytes = (uint64_t)half * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

    ds4_gpu_tensor *x_lo = ds4_gpu_tensor_alloc(half_bytes);
    ds4_gpu_tensor *x_hi = ds4_gpu_tensor_alloc(half_bytes);
    ds4_gpu_tensor *out_lo = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *out_hi = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x_lo != NULL); TEST_ASSERT(x_hi != NULL);
    TEST_ASSERT(out_lo != NULL); TEST_ASSERT(out_hi != NULL);
    if (!x_lo || !x_hi || !out_lo || !out_hi) {
        ds4_gpu_tensor_free(x_lo); ds4_gpu_tensor_free(x_hi);
        ds4_gpu_tensor_free(out_lo); ds4_gpu_tensor_free(out_hi);
        free(weights_raw); return;
    }

    float x_host[128];
    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 7u + (i ^ 13u)) % 71u) - 35;
        x_host[i] = (float)v / 80.0f;
    }
    float out_lo_host[64], out_hi_host[64];

    TEST_ASSERT(ds4_gpu_tensor_write(x_lo, 0, x_host, half_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(x_hi, 0, x_host + half, half_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);

    /* peer-low: in-blocks [0, half/32); weight base unshifted; x = x_full[0:half). */
    TEST_ASSERT(ds4_gpu_matmul_q8_0_rowslice_tensor(out_lo, weights_raw, weight_alloc,
                                                    0, in_dim, half, out_dim, x_lo) != 0);
    /* peer-high: in-blocks [half/32, in_dim/32); weight base + (half/32)*34; x = x_full[half:]. */
    TEST_ASSERT(ds4_gpu_matmul_q8_0_rowslice_tensor(out_hi, weights_raw, weight_alloc,
                                                    (uint64_t)(half / 32u) * 34u,
                                                    in_dim, half, out_dim, x_hi) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out_lo, 0, out_lo_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out_hi, 0, out_hi_host, out_bytes) != 0);

    float max_abs = 0.0f, rms = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        const uint8_t *wr = weights + (uint64_t)o * row_bytes;
        float ref = 0.0f;
        for (uint32_t b = 0; b < in_dim / 32u; b++) {
            uint16_t sb; memcpy(&sb, wr + b * 34u, sizeof(sb));
            const float scale = test_f16_to_f32(sb);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++)
                ref += scale * (float)qs[i] * x_host[b * 32u + i];
        }
        const float got = out_lo_host[o] + out_hi_host[o];   /* the host all-reduce sum */
        TEST_ASSERT(isfinite(got));
        const float err = fabsf(got - ref);
        if (err > max_abs) max_abs = err;
        rms += err * err;
    }
    rms = sqrtf(rms / (float)out_dim);
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    ds4_gpu_tensor_free(x_lo); ds4_gpu_tensor_free(x_hi);
    ds4_gpu_tensor_free(out_lo); ds4_gpu_tensor_free(out_hi);
    free(weights_raw);
}

/* ---- go1b strict-1-bit routed-MoE numeric regression -----------------------
 *
 * The go1b model first ran but produced a clean repetition loop: in OFFLOAD mode
 * (over-budget model, non-resident expert mmap views) the routed experts read as
 * ZEROS because go1b skipped the A3 CPU-gather that q2 uses. This test builds a
 * synthetic block_go1b routed-expert model (gate/up/down), forces the SAME offload
 * CPU-gather-to-resident-scratch + mm_id dispatch the real 42 GiB model runs
 * (DS4_METAL_EXPERT_OFFLOAD=1), and compares the routed-MoE output to a CPU
 * reference (dequant +/-d per sign bit, gate/up matmul, SwiGLU, route weight, down
 * matmul, sum over the 6 experts). It catches go1b dequant element-ordering, the
 * row/expert byte strides ((ncols/256)*34), expert-id indexing, half-scale read,
 * and -- crucially -- the gather dispatch: if the experts are not gathered into
 * resident scratch the kernel reads garbage/zeros and the output misses the
 * (non-trivial) reference by ~100%.
 *
 * NB: true page-reclamation non-residency only occurs with a real over-budget
 * model; a small resident synthetic buffer reads correctly on the GPU regardless
 * of residency-set membership. This isolates the go1b kernel + gather MATH through
 * the offload code path; the end-to-end non-residency fix is verified by running
 * the 42 GiB model (loop -> varied tokens). */
#define TEST_GO1B_GGML_TYPE 40u   /* DS4_TENSOR_GO1B / DS4_METAL_TENSOR_GO1B */

/* Per-row fp16 scale d, round-tripped through f16 so the reference reads exactly
 * what dequantize_go1b reads (half d -> +d / -d). */
static float test_go1b_scale(uint32_t tag, uint32_t e, uint32_t r) {
    uint32_t h = tag * 0x9E3779B1u + e * 0x85EBCA77u + r * 0xC2B2AE3Du + 0x165667B1u;
    h ^= h >> 15;
    return test_f16_to_f32(test_float_to_f16(0.10f + (float)(h % 21u) * 0.005f));
}
static uint32_t test_go1b_sign(uint32_t tag, uint32_t e, uint32_t r, uint32_t c) {
    uint32_t h = tag * 0x27D4EB2Fu + e * 0x9E3779B1u + r * 0x85EBCA77u + c * 0xC2B2AE3Du;
    h ^= h >> 13; h *= 0x5BD1E995u; h ^= h >> 15;
    return (h >> 7) & 1u;   /* 1 -> +d, 0 -> -d */
}
static float test_go1b_weight(uint32_t tag, uint32_t e, uint32_t r, uint32_t c) {
    const float d = test_go1b_scale(tag, e, r);
    return test_go1b_sign(tag, e, r, c) ? d : -d;
}
/* Write one go1b expert (rows x cols, cols % 256 == 0) at base. Each row repeats
 * its fp16 scale into every 256-block and packs 256 sign bits per block. */
static void test_fill_go1b_expert(uint8_t *base, uint32_t tag, uint32_t e,
                                  uint32_t rows, uint32_t cols, uint64_t row_bytes) {
    const uint32_t nblk = cols / 256u;
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t d16 = test_float_to_f16(test_go1b_scale(tag, e, r));
        uint8_t *row = base + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < nblk; b++) {
            uint8_t *blk = row + (uint64_t)b * 34u;
            memcpy(blk, &d16, sizeof(d16));
            uint8_t *signs = blk + 2u;
            memset(signs, 0, 32u);
            for (uint32_t i = 0; i < 256u; i++) {
                const uint32_t c = b * 256u + i;
                if (test_go1b_sign(tag, e, r, c)) signs[i >> 3] |= (uint8_t)(1u << (i & 7u));
            }
        }
    }
}

void test_metal_go1b_routed_moe(void) {
    /* Exercise the over-budget OFFLOAD dispatch (CPU gather -> resident scratch ->
     * mm_id). pread self-disables without a model fd, so the synthetic
     * host-buffer model is gathered by memcpy. */
    ds4_gpu_set_expert_offload(1);

    /* Real-model shapes: multi-block rows (in_dim=2048 -> 8 go1b blocks/row,
     * mid_dim=2048 -> 8 blocks) so the test exercises the per-block scale/sign
     * advance the 42 GiB model uses, not just a single block. */
    const uint32_t in_dim = 2048, mid_dim = 2048, out_dim = 2048;
    const uint32_t n_total = 8, n_sel = 6;                      /* mm_id map0 supports 6 */
    const uint64_t blk = 34u;                                   /* sizeof(block_go1b) */
    const uint64_t gate_row_bytes = (uint64_t)(in_dim / 256u) * blk;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)(mid_dim / 256u) * blk;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_total * gate_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total * down_expert_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off   = gate_tensor_bytes;
    const uint64_t down_off = gate_tensor_bytes * 2u;
    const uint64_t model_bytes = down_off + down_tensor_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    if (!model_raw) return;
    uint8_t *model = model_raw;
    memset(model, 0, (size_t)model_alloc);
    for (uint32_t e = 0; e < n_total; e++) {
        test_fill_go1b_expert(model + gate_off + (uint64_t)e * gate_expert_bytes, 0, e, mid_dim, in_dim, gate_row_bytes);
        test_fill_go1b_expert(model + up_off   + (uint64_t)e * gate_expert_bytes, 1, e, mid_dim, in_dim, gate_row_bytes);
        test_fill_go1b_expert(model + down_off + (uint64_t)e * down_expert_bytes, 2, e, out_dim, mid_dim, down_row_bytes);
    }

    const uint32_t sel[6] = {0u, 2u, 4u, 6u, 1u, 3u};   /* 6 distinct of 8 */
    int32_t sel_i[6];
    float   rw[6];
    for (uint32_t s = 0; s < n_sel; s++) { sel_i[s] = (int32_t)sel[s]; rw[s] = 0.4f + 0.1f * (float)s; }

    float *x_host  = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    float *ref     = calloc(out_dim, sizeof(float));
    float *mref    = malloc((size_t)mid_dim * sizeof(float));
    ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_sel * mid_dim * sizeof(float));
    ds4_gpu_tensor *up  = ds4_gpu_tensor_alloc((uint64_t)n_sel * mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_sel * mid_dim * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc((uint64_t)n_sel * out_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc((uint64_t)n_sel * sizeof(int32_t));
    ds4_gpu_tensor *wt_t  = ds4_gpu_tensor_alloc((uint64_t)n_sel * sizeof(float));

    if (x_host && out_host && ref && mref && x && gate && up && mid && experts && out && sel_t && wt_t) {
        for (uint32_t c = 0; c < in_dim; c++)
            x_host[c] = (float)((int)((c * 7u + 3u) % 29u) - 14) / 35.0f;   /* ~[-0.4, 0.4] */

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sel_t, 0, sel_i, sizeof(sel_i)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(wt_t, 0, rw, sizeof(rw)) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);   /* the real inference path: fused SwiGLU, f16 mid */

        bool mid_f16 = false;
        const int ok = ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, experts,
            NULL /* residual sidecar: none in this kernel test */,
            model_raw, model_alloc, gate_off, up_off, down_off,
            TEST_GO1B_GGML_TYPE, TEST_GO1B_GGML_TYPE,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, sel_t, wt_t, n_total, n_sel,
            0.0f /* clamp off */, x, 0 /* layer */, 1 /* n_tokens */,
            0, 0, &mid_f16);
        TEST_ASSERT(ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

        /* CPU reference: pair p uses expert sel[p] and route weight rw[p]. */
        for (uint32_t s = 0; s < n_sel; s++) {
            const uint32_t e = sel[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                float gv = 0.0f, uv = 0.0f;
                for (uint32_t c = 0; c < in_dim; c++) {
                    gv += test_go1b_weight(0, e, r, c) * x_host[c];
                    uv += test_go1b_weight(1, e, r, c) * x_host[c];
                }
                const float silu = gv / (1.0f + expf(-gv));
                mref[r] = silu * uv * rw[s];
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                float ov = 0.0f;
                for (uint32_t c = 0; c < mid_dim; c++)
                    ov += test_go1b_weight(2, e, r, c) * mref[c];
                ref[r] += ov;
            }
        }

        float max_abs = 0.0f, sq = 0.0f, ref_sq = 0.0f;
        for (uint32_t r = 0; r < out_dim; r++) {
            TEST_ASSERT(isfinite(out_host[r]));
            const float err = fabsf(out_host[r] - ref[r]);
            if (err > max_abs) max_abs = err;
            sq += err * err;
            ref_sq += ref[r] * ref[r];
        }
        const float rms = sqrtf(sq / (float)out_dim);
        const float ref_rms = sqrtf(ref_sq / (float)out_dim);
        fprintf(stderr, "ds4: go1b routed-MoE test ref_rms=%.4f rms_err=%.5f max_abs=%.5f (rel=%.4f)\n",
                ref_rms, rms, max_abs, ref_rms > 0 ? rms / ref_rms : 0.0f);
        /* A non-gathered (zeroed) expert read would leave out ~ 0 -> rel ~ 1.0. */
        TEST_ASSERT(ref_rms > 0.5f);
        TEST_ASSERT(rms < 0.06f * ref_rms);
        TEST_ASSERT(max_abs < 0.25f * ref_rms);
    } else {
        TEST_ASSERT(0 && "go1b routed-MoE test allocation failed");
    }

    free(x_host); free(out_host); free(ref); free(mref);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(experts); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(sel_t); ds4_gpu_tensor_free(wt_t);
    free(model_raw);
}

/* ---- go2b (2-bit ±d1±d2) routed-MoE cross-GPU numeric regression -----------
 * Mirrors the go1b test but with the go2b 68-byte block {d1 f16, d2 f16, s1[32],
 * s2[32]}, value = (s1?+d1:-d1)+(s2?+d2:-d2). Purpose: measure whether the go2b
 * dequant+matmul is as cross-GPU consistent as go1b (rel 0.0006). If go2b rel
 * error differs across M1/M4 while go1b matches, the monolithic mixed model's
 * M1-vs-M4 generation divergence localizes to the go2b kernel. */
#define TEST_GO2B_GGML_TYPE 41u
static float test_go2b_d1(uint32_t tag, uint32_t e, uint32_t r) {
    uint32_t h = tag * 0x9E3779B1u + e * 0x85EBCA77u + r * 0xC2B2AE3Du + 0x165667B1u;
    h ^= h >> 15;
    return test_f16_to_f32(test_float_to_f16(0.10f + (float)(h % 21u) * 0.005f));
}
static float test_go2b_d2(uint32_t tag, uint32_t e, uint32_t r) {
    uint32_t h = tag * 0x27D4EB2Fu + e * 0xC2B2AE3Du + r * 0x9E3779B1u + 0x27220A95u;
    h ^= h >> 14;
    return test_f16_to_f32(test_float_to_f16(0.03f + (float)(h % 15u) * 0.004f));
}
static uint32_t test_go2b_s1(uint32_t tag, uint32_t e, uint32_t r, uint32_t c) {
    uint32_t h = tag * 0x27D4EB2Fu + e * 0x9E3779B1u + r * 0x85EBCA77u + c * 0xC2B2AE3Du;
    h ^= h >> 13; h *= 0x5BD1E995u; h ^= h >> 15;
    return (h >> 7) & 1u;
}
static uint32_t test_go2b_s2(uint32_t tag, uint32_t e, uint32_t r, uint32_t c) {
    uint32_t h = tag * 0x85EBCA77u + e * 0xC2B2AE3Du + r * 0x27D4EB2Fu + c * 0x9E3779B1u;
    h ^= h >> 12; h *= 0x2545F491u; h ^= h >> 14;
    return (h >> 9) & 1u;
}
static float test_go2b_weight(uint32_t tag, uint32_t e, uint32_t r, uint32_t c) {
    const float d1 = test_go2b_d1(tag, e, r), d2 = test_go2b_d2(tag, e, r);
    const float a = test_go2b_s1(tag, e, r, c) ? d1 : -d1;
    const float b = test_go2b_s2(tag, e, r, c) ? d2 : -d2;
    return a + b;
}
static void test_fill_go2b_expert(uint8_t *base, uint32_t tag, uint32_t e,
                                  uint32_t rows, uint32_t cols, uint64_t row_bytes) {
    const uint32_t nblk = cols / 256u;
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t d1_16 = test_float_to_f16(test_go2b_d1(tag, e, r));
        const uint16_t d2_16 = test_float_to_f16(test_go2b_d2(tag, e, r));
        uint8_t *row = base + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < nblk; b++) {
            uint8_t *blk = row + (uint64_t)b * 68u;
            memcpy(blk, &d1_16, 2u); memcpy(blk + 2u, &d2_16, 2u);
            uint8_t *s1 = blk + 4u, *s2 = blk + 36u;
            memset(s1, 0, 32u); memset(s2, 0, 32u);
            for (uint32_t i = 0; i < 256u; i++) {
                const uint32_t c = b * 256u + i;
                if (test_go2b_s1(tag, e, r, c)) s1[i >> 3] |= (uint8_t)(1u << (i & 7u));
                if (test_go2b_s2(tag, e, r, c)) s2[i >> 3] |= (uint8_t)(1u << (i & 7u));
            }
        }
    }
}
void test_metal_go2b_routed_moe(void) {
    ds4_gpu_set_expert_offload(1);
    const uint32_t in_dim = 2048, mid_dim = 2048, out_dim = 2048;
    const uint32_t n_total = 8, n_sel = 6;
    const uint64_t blk = 68u;   /* sizeof(block_go2b) */
    const uint64_t gate_row_bytes = (uint64_t)(in_dim / 256u) * blk;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)(mid_dim / 256u) * blk;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_tensor_bytes = (uint64_t)n_total * gate_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total * down_expert_bytes;
    const uint64_t gate_off = 0, up_off = gate_tensor_bytes, down_off = gate_tensor_bytes * 2u;
    const uint64_t model_bytes = down_off + down_tensor_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    if (!model_raw) return;
    uint8_t *model = model_raw; memset(model, 0, (size_t)model_alloc);
    for (uint32_t e = 0; e < n_total; e++) {
        test_fill_go2b_expert(model + gate_off + (uint64_t)e * gate_expert_bytes, 0, e, mid_dim, in_dim, gate_row_bytes);
        test_fill_go2b_expert(model + up_off   + (uint64_t)e * gate_expert_bytes, 1, e, mid_dim, in_dim, gate_row_bytes);
        test_fill_go2b_expert(model + down_off + (uint64_t)e * down_expert_bytes, 2, e, out_dim, mid_dim, down_row_bytes);
    }
    const uint32_t sel[6] = {0u, 2u, 4u, 6u, 1u, 3u};
    int32_t sel_i[6]; float rw[6];
    for (uint32_t s = 0; s < n_sel; s++) { sel_i[s] = (int32_t)sel[s]; rw[s] = 0.4f + 0.1f * (float)s; }
    float *x_host = malloc((size_t)in_dim * sizeof(float)), *out_host = malloc((size_t)out_dim * sizeof(float));
    float *ref = calloc(out_dim, sizeof(float)), *mref = malloc((size_t)mid_dim * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_sel * mid_dim * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n_sel * mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_sel * mid_dim * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc((uint64_t)n_sel * out_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc((uint64_t)n_sel * sizeof(int32_t));
    ds4_gpu_tensor *wt_t = ds4_gpu_tensor_alloc((uint64_t)n_sel * sizeof(float));
    if (x_host && out_host && ref && mref && x && gate && up && mid && experts && out && sel_t && wt_t) {
        for (uint32_t c = 0; c < in_dim; c++)
            x_host[c] = (float)((int)((c * 7u + 3u) % 29u) - 14) / 35.0f;
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sel_t, 0, sel_i, sizeof(sel_i)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(wt_t, 0, rw, sizeof(rw)) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);
        bool mid_f16 = false;
        const int ok = ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, experts, NULL, model_raw, model_alloc, gate_off, up_off, down_off,
            TEST_GO2B_GGML_TYPE, TEST_GO2B_GGML_TYPE,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, sel_t, wt_t, n_total, n_sel, 0.0f, x, 0, 1, 0, 0, &mid_f16);
        TEST_ASSERT(ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);
        for (uint32_t s = 0; s < n_sel; s++) {
            const uint32_t e = sel[s];
            for (uint32_t r = 0; r < mid_dim; r++) {
                float gv = 0.0f, uv = 0.0f;
                for (uint32_t c = 0; c < in_dim; c++) {
                    gv += test_go2b_weight(0, e, r, c) * x_host[c];
                    uv += test_go2b_weight(1, e, r, c) * x_host[c];
                }
                const float silu = gv / (1.0f + expf(-gv));
                mref[r] = silu * uv * rw[s];
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                float ov = 0.0f;
                for (uint32_t c = 0; c < mid_dim; c++) ov += test_go2b_weight(2, e, r, c) * mref[c];
                ref[r] += ov;
            }
        }
        float max_abs = 0.0f, sq = 0.0f, ref_sq = 0.0f;
        for (uint32_t r = 0; r < out_dim; r++) {
            TEST_ASSERT(isfinite(out_host[r]));
            const float err = fabsf(out_host[r] - ref[r]);
            if (err > max_abs) max_abs = err; sq += err * err; ref_sq += ref[r] * ref[r];
        }
        const float rms = sqrtf(sq / (float)out_dim), ref_rms = sqrtf(ref_sq / (float)out_dim);
        fprintf(stderr, "ds4: go2b routed-MoE test ref_rms=%.4f rms_err=%.5f max_abs=%.5f (rel=%.4f)\n",
                ref_rms, rms, max_abs, ref_rms > 0 ? rms / ref_rms : 0.0f);
        TEST_ASSERT(ref_rms > 0.3f);
        TEST_ASSERT(rms < 0.06f * ref_rms);
    } else { TEST_ASSERT(0 && "go2b routed-MoE test allocation failed"); }
    free(x_host); free(out_host); free(ref); free(mref);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(experts); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(sel_t); ds4_gpu_tensor_free(wt_t);
    free(model_raw);
}

#endif /* !DS4_NO_GPU */
typedef int ds4_t_metal_kernels_moe_nonempty_tu; /* 空TU防御(CPU构建) */
