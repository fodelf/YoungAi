/* t_metal_kernels_group.c — Metal kernel 数值回归后半: corr_apply + kernel 组入口 + ratio-4 短 prefill (机械拆分自 tests/ds4_test.c, 重构阶段8)。 */
#include "test_internal.h"
#ifndef DS4_NO_GPU

/* go1b "hidden variable z^L" four-loss correction numeric regression.
 *
 * Builds synthetic U,V,C,b,beta,delta + a known x and a known 1-bit baseline o_hat,
 * then checks that:
 *   (1) kernel_dsv4_corr_apply reproduces an independent CPU reference of
 *       out += sum over selected e of  U @ (C[e] .* (V @ x)) + b + beta[e];
 *   (2) kernel_dsv4_corr_router_bias reproduces logits[t][e] += delta[e].
 * Multi-token (n_tokens>1) exercises the per-token threadgroup dispatch. */
static void test_metal_corr_apply(void) {
    const uint32_t d_model = 320, d_l = 24, n_expert = 40, n_sel = 6, n_tok = 3;

    float *U = malloc((size_t)d_model * d_l * sizeof(float));   /* [d_model][d_l] */
    float *V = malloc((size_t)d_l * d_model * sizeof(float));   /* [d_l][d_model] */
    float *C = malloc((size_t)n_expert * d_l * sizeof(float));  /* [n_expert][d_l] */
    float *b = malloc((size_t)d_model * sizeof(float));
    float *beta = malloc((size_t)n_expert * sizeof(float));
    float *delta = malloc((size_t)n_expert * sizeof(float));
    float *x = malloc((size_t)n_tok * d_model * sizeof(float));
    int32_t *sel = malloc((size_t)n_tok * n_sel * sizeof(int32_t));
    float *base = malloc((size_t)n_tok * d_model * sizeof(float));  /* o_hat baseline */
    float *ref = malloc((size_t)n_tok * d_model * sizeof(float));
    float *got = malloc((size_t)n_tok * d_model * sizeof(float));
    TEST_ASSERT(U && V && C && b && beta && delta && x && sel && base && ref && got);

    for (uint32_t k = 0; k < d_model * d_l; k++) U[k] = (float)((int)((k * 7u + 1u) % 17u) - 8) / 32.0f;
    for (uint32_t k = 0; k < d_l * d_model; k++) V[k] = (float)((int)((k * 5u + 3u) % 19u) - 9) / 48.0f;
    for (uint32_t k = 0; k < n_expert * d_l; k++) C[k] = (float)((int)((k * 11u + 2u) % 13u) - 6) / 24.0f;
    for (uint32_t d = 0; d < d_model; d++) b[d] = (float)((int)((d * 3u) % 7u) - 3) / 40.0f;
    for (uint32_t e = 0; e < n_expert; e++) beta[e] = (float)((int)((e * 9u + 1u) % 11u) - 5) / 50.0f;
    for (uint32_t e = 0; e < n_expert; e++) delta[e] = (float)((int)((e * 13u + 4u) % 23u) - 11) / 16.0f;
    for (uint32_t k = 0; k < n_tok * d_model; k++) x[k] = (float)((int)((k * 17u + 5u) % 29u) - 14) / 35.0f;
    for (uint32_t k = 0; k < n_tok * d_model; k++) base[k] = (float)((int)((k * 23u + 7u) % 31u) - 15) / 20.0f;
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t s = 0; s < n_sel; s++)
            sel[t * n_sel + s] = (int32_t)((t * 5u + s * 7u + 1u) % n_expert);   /* in-range ids */

    /* Independent CPU reference, literal from the spec. */
    float *vx = malloc((size_t)d_l * sizeof(float));
    TEST_ASSERT(vx != NULL);
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *xt = x + (size_t)t * d_model;
        for (uint32_t i = 0; i < d_l; i++) {
            float a = 0.0f;
            for (uint32_t j = 0; j < d_model; j++) a += V[(size_t)i * d_model + j] * xt[j];
            vx[i] = a;
        }
        float *rt = ref + (size_t)t * d_model;
        for (uint32_t d = 0; d < d_model; d++) rt[d] = base[(size_t)t * d_model + d];
        for (uint32_t s = 0; s < n_sel; s++) {
            const uint32_t e = (uint32_t)sel[t * n_sel + s];
            for (uint32_t d = 0; d < d_model; d++) {
                float p = 0.0f;
                for (uint32_t i = 0; i < d_l; i++) p += U[(size_t)d * d_l + i] * C[(size_t)e * d_l + i] * vx[i];
                rt[d] += p + b[d] + beta[e];
            }
        }
    }

    ds4_gpu_tensor *gU = ds4_gpu_tensor_alloc((uint64_t)d_model * d_l * sizeof(float));
    ds4_gpu_tensor *gV = ds4_gpu_tensor_alloc((uint64_t)d_l * d_model * sizeof(float));
    ds4_gpu_tensor *gC = ds4_gpu_tensor_alloc((uint64_t)n_expert * d_l * sizeof(float));
    ds4_gpu_tensor *gb = ds4_gpu_tensor_alloc((uint64_t)d_model * sizeof(float));
    ds4_gpu_tensor *gbeta = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *gdelta = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc((uint64_t)n_tok * d_model * sizeof(float));
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)n_tok * n_sel * sizeof(int32_t));
    ds4_gpu_tensor *gout = ds4_gpu_tensor_alloc((uint64_t)n_tok * d_model * sizeof(float));
    TEST_ASSERT(gU && gV && gC && gb && gbeta && gdelta && gx && gsel && gout);

    if (gU && gV && gC && gb && gbeta && gdelta && gx && gsel && gout) {
        TEST_ASSERT(ds4_gpu_tensor_write(gU, 0, U, (uint64_t)d_model * d_l * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gV, 0, V, (uint64_t)d_l * d_model * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gC, 0, C, (uint64_t)n_expert * d_l * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gb, 0, b, (uint64_t)d_model * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gbeta, 0, beta, (uint64_t)n_expert * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gx, 0, x, (uint64_t)n_tok * d_model * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)n_tok * n_sel * sizeof(int32_t)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(gout, 0, base, (uint64_t)n_tok * d_model * sizeof(float)) != 0);

        TEST_ASSERT(ds4_gpu_corr_apply(gout, gx, gU, gV, gC, gb, gbeta, gsel,
                                       d_model, d_l, n_expert, n_sel, n_tok) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(gout, 0, got, (uint64_t)n_tok * d_model * sizeof(float)) != 0);

        float max_abs = 0.0f, sq = 0.0f, ref_sq = 0.0f;
        for (uint32_t k = 0; k < n_tok * d_model; k++) {
            TEST_ASSERT(isfinite(got[k]));
            const float err = fabsf(got[k] - ref[k]);
            if (err > max_abs) max_abs = err;
            sq += err * err;
            ref_sq += ref[k] * ref[k];
        }
        const float rel = ref_sq > 0.0f ? sqrtf(sq / ref_sq) : 0.0f;
        fprintf(stderr, "ds4: corr_apply test ref_rms=%.4f max_abs=%.6f rel=%.7f\n",
                sqrtf(ref_sq / (float)(n_tok * d_model)), max_abs, rel);
        TEST_ASSERT(ref_sq > 0.0f);
        TEST_ASSERT(rel < 1.0e-3f);

        /* Router bias: logits[t][e] += delta[e]. */
        float *logit = malloc((size_t)n_tok * n_expert * sizeof(float));
        float *logit_ref = malloc((size_t)n_tok * n_expert * sizeof(float));
        float *logit_got = malloc((size_t)n_tok * n_expert * sizeof(float));
        TEST_ASSERT(logit && logit_ref && logit_got);
        ds4_gpu_tensor *glogit = ds4_gpu_tensor_alloc((uint64_t)n_tok * n_expert * sizeof(float));
        TEST_ASSERT(glogit != NULL);
        if (logit && logit_ref && logit_got && glogit) {
            for (uint32_t k = 0; k < n_tok * n_expert; k++) logit[k] = (float)((int)((k * 19u + 2u) % 37u) - 18) / 12.0f;
            for (uint32_t t = 0; t < n_tok; t++)
                for (uint32_t e = 0; e < n_expert; e++)
                    logit_ref[t * n_expert + e] = logit[t * n_expert + e] + delta[e];
            TEST_ASSERT(ds4_gpu_tensor_write(glogit, 0, logit, (uint64_t)n_tok * n_expert * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(gdelta, 0, delta, (uint64_t)n_expert * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_corr_router_bias(glogit, gdelta, n_expert, n_tok) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(glogit, 0, logit_got, (uint64_t)n_tok * n_expert * sizeof(float)) != 0);
            float bmax = 0.0f;
            for (uint32_t k = 0; k < n_tok * n_expert; k++) {
                const float err = fabsf(logit_got[k] - logit_ref[k]);
                if (err > bmax) bmax = err;
            }
            fprintf(stderr, "ds4: corr_router_bias test max_abs=%.7f\n", bmax);
            TEST_ASSERT(bmax < 1.0e-5f);
        }
        free(logit); free(logit_ref); free(logit_got);
        ds4_gpu_tensor_free(glogit);

        /* Store-variant parity: corr_delta writes the raw correction term;
         * base[k] + delta[k] must reproduce the in-place result BITWISE (same
         * fadd operands — this is the contract the fused shared-down consumer
         * relies on). */
        if (ds4_gpu_corr_delta_supported()) {
            float *ddelta = malloc((size_t)n_tok * d_model * sizeof(float));
            ds4_gpu_tensor *gdelta_out = ds4_gpu_tensor_alloc((uint64_t)n_tok * d_model * sizeof(float));
            TEST_ASSERT(ddelta && gdelta_out);
            if (ddelta && gdelta_out) {
                TEST_ASSERT(ds4_gpu_corr_apply_delta(gdelta_out, gx, gU, gV, gC, gb, gbeta, gsel,
                                                     d_model, d_l, n_expert, n_sel, n_tok) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(gdelta_out, 0, ddelta,
                                                (uint64_t)n_tok * d_model * sizeof(float)) != 0);
                uint32_t mism = 0;
                for (uint32_t k = 0; k < n_tok * d_model; k++) {
                    const float recon = base[k] + ddelta[k];
                    if (recon != got[k]) mism++;
                }
                fprintf(stderr, "ds4: corr_delta store-variant bitwise mismatches=%u\n", mism);
                TEST_ASSERT(mism == 0);
            }
            free(ddelta);
            ds4_gpu_tensor_free(gdelta_out);
        }
    }

    free(vx);
    ds4_gpu_tensor_free(gU); ds4_gpu_tensor_free(gV); ds4_gpu_tensor_free(gC);
    ds4_gpu_tensor_free(gb); ds4_gpu_tensor_free(gbeta); ds4_gpu_tensor_free(gdelta);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gout);
    free(U); free(V); free(C); free(b); free(beta); free(delta);
    free(x); free(sel); free(base); free(ref); free(got);
}

void test_metal_kernel_group(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_q8_0_prefill_matmul_unaligned();
    test_metal_q8_0_rowslice_tp();
    test_metal_go1b_routed_moe();
    test_metal_go2b_routed_moe();
    test_metal_corr_apply();
}

void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}


#endif /* !DS4_NO_GPU */
typedef int ds4_t_metal_kernels_group_nonempty_tu; /* 空TU防御(CPU构建) */
