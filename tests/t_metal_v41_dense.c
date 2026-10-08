/* t_metal_v41_dense.c — V4.1 Metal 稠密线性层回归(2026-10-08): 五种权重类型的 GEMV(n=1/3)与 GEMM(n=40)、分组块对角、转置乘(反传)、
 * f32 GEMM 四种转置、嵌入取行、列平方和、放大器应用。金标 = src/common 解码 + CPU 逐式乘。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_quantfmt.h"
#include "../src/metal/metal_v41_args.h"
#include "../ds4_gpu_v41.h"
#include "../ds4_gpu_bwd.h"

static const char *wt_name(uint32_t wt) { static const char *n[] = { "fp4x32", "q4_K", "bf16", "f32", "fp8blk" }; return n[wt]; }
static uint32_t wt_ggt(uint32_t wt) { static const uint32_t g[] = { DS4_GGT_FP4X32, DS4_GGT_Q4_K, DS4_GGT_BF16, DS4_GGT_F32, DS4_GGT_FP8_32X32 }; return g[wt]; }

/* 契约入口按类型分发 */
static int call_matmul(uint32_t wt, ds4_gpu_tensor *out, uint64_t off, uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    const void *m = test_v41_model_map(); const uint64_t sz = test_v41_model_size();
    switch (wt) {
        case V41_WT_FP4X32: return ds4_gpu_v41_matmul_fp4x32_tensor(out, m, sz, off, in_dim, out_dim, x, n_tok, round_out);
        case V41_WT_Q4K: return ds4_gpu_v41_matmul_q4k_tensor(out, m, sz, off, in_dim, out_dim, x, n_tok, round_out);
        case V41_WT_BF16: return ds4_gpu_v41_matmul_bf16_tensor(out, m, sz, off, in_dim, out_dim, x, n_tok);
        case V41_WT_F32: return ds4_gpu_v41_matmul_f32_tensor(out, m, sz, off, in_dim, out_dim, x, n_tok);
        default: return ds4_gpu_v41_matmul_fp8blk_round_tensor(out, m, sz, off, in_dim, out_dim, x, n_tok, round_out);
    }
}
static int call_grouped(uint32_t wt, ds4_gpu_tensor *low, uint64_t off, uint32_t G, uint32_t gd, uint32_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    const void *m = test_v41_model_map(); const uint64_t sz = test_v41_model_size();
    switch (wt) {
        case V41_WT_FP4X32: return ds4_gpu_v41_grouped_matmul_fp4x32_tensor(low, m, sz, off, G, gd, rank, heads, n_tok, round_out);
        case V41_WT_Q4K: return ds4_gpu_v41_grouped_matmul_q4k_tensor(low, m, sz, off, G, gd, rank, heads, n_tok, round_out);
        case V41_WT_FP8BLK: return ds4_gpu_v41_grouped_matmul_fp8blk_tensor(low, m, sz, off, G, gd, rank, heads, n_tok, round_out);
        default: return -1;
    }
}

static void test_dense_type(uint32_t wt) {
    const uint32_t in_dim = 512, out_dim = 96;   /* in 是 256 的倍数(q4_K), out 不是 8 的倍数(尾行) */
    const uint64_t wb = test_v41_wbytes(wt, out_dim, in_dim);
    TEST_ASSERT(test_v41_model_begin(wb + 4096));
    const uint64_t off = test_v41_model_alloc(wb);
    float *wref = malloc((size_t)out_dim * in_dim * 4);
    test_v41_make_weights(wt, out_dim, in_dim, test_v41_model_ptr(off), wref);
    TEST_ASSERT(test_v41_map());
    const uint32_t ntoks[3] = { 1, 3, 40 };
    for (int c = 0; c < 3; c++) {
        const uint32_t n = ntoks[c];
        const int round_out = (wt == V41_WT_FP4X32 || wt == V41_WT_Q4K || wt == V41_WT_FP8BLK) ? (int)(c & 1) : 0;
        float *x = malloc((size_t)n * in_dim * 4), *ref = malloc((size_t)n * out_dim * 4), *got = malloc((size_t)n * out_dim * 4);
        test_v41_fill_bf16(x, (uint64_t)n * in_dim, 1.0f);
        ds4_gpu_tensor *tx = test_v41_tensor(x, (uint64_t)n * in_dim * 4), *to = test_v41_tensor(NULL, (uint64_t)n * out_dim * 4);
        TEST_ASSERT(call_matmul(wt, to, off, in_dim, out_dim, tx, n, round_out) != 0);
        TEST_ASSERT(test_v41_read(to, got, (uint64_t)n * out_dim * 4));
        test_v41_ref_matmul(x, wref, ref, n, out_dim, in_dim, round_out);
        char name[64]; snprintf(name, sizeof name, "%s matmul n=%u r=%d", wt_name(wt), n, round_out);
        test_v41_cmp(name, got, ref, (uint64_t)n * out_dim, round_out ? 6e-3f : 1e-4f, 1e-3f);
        /* 反传转置乘: gx[n][in] = gy[n][out]·W(以 got 当 gy, 顺手验 accumulate) */
        {
            float *gx = malloc((size_t)n * in_dim * 4), *gref = malloc((size_t)n * in_dim * 4);
            for (uint64_t i = 0; i < (uint64_t)n * in_dim; i++) gx[i] = 1.0f;
            ds4_gpu_tensor *tgx = test_v41_tensor(gx, (uint64_t)n * in_dim * 4);
            TEST_ASSERT(ds4_gpu_bwd_matmul_t_tensor(tgx, test_v41_model_map(), test_v41_model_size(), wt_ggt(wt), off, in_dim, out_dim, to, n, 1) != 0);
            TEST_ASSERT(test_v41_read(tgx, gx, (uint64_t)n * in_dim * 4));
            for (uint32_t t = 0; t < n; t++) for (uint32_t k = 0; k < in_dim; k++) {
                double s = 1.0;
                for (uint32_t o = 0; o < out_dim; o++) s += (double)got[(uint64_t)t * out_dim + o] * wref[(uint64_t)o * in_dim + k];
                gref[(uint64_t)t * in_dim + k] = (float)s;
            }
            snprintf(name, sizeof name, "%s matmul_t n=%u", wt_name(wt), n);
            test_v41_cmp(name, gx, gref, (uint64_t)n * in_dim, 2e-4f, 1e-3f);
            ds4_gpu_tensor_free(tgx); free(gx); free(gref);
        }
        ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(to); free(x); free(ref); free(got);
    }
    free(wref);
}

static void test_grouped_type(uint32_t wt) {
    const uint32_t G = 4, gd = 256, rank = 64, in_all = G * gd, out_all = G * rank;
    const uint64_t wb = test_v41_wbytes(wt, out_all, gd);
    TEST_ASSERT(test_v41_model_begin(wb + 4096));
    const uint64_t off = test_v41_model_alloc(wb);
    float *wref = malloc((size_t)out_all * gd * 4);
    test_v41_make_weights(wt, out_all, gd, test_v41_model_ptr(off), wref);
    TEST_ASSERT(test_v41_map());
    const uint32_t ntoks[2] = { 2, 24 };
    for (int c = 0; c < 2; c++) {
        const uint32_t n = ntoks[c];
        float *x = malloc((size_t)n * in_all * 4), *ref = malloc((size_t)n * out_all * 4), *got = malloc((size_t)n * out_all * 4);
        test_v41_fill_bf16(x, (uint64_t)n * in_all, 1.0f);
        ds4_gpu_tensor *tx = test_v41_tensor(x, (uint64_t)n * in_all * 4), *to = test_v41_tensor(NULL, (uint64_t)n * out_all * 4);
        TEST_ASSERT(call_grouped(wt, to, off, G, gd, rank, tx, n, 1) != 0);
        TEST_ASSERT(test_v41_read(to, got, (uint64_t)n * out_all * 4));
        for (uint32_t t = 0; t < n; t++) for (uint32_t g = 0; g < G; g++) for (uint32_t r = 0; r < rank; r++) {
            double s = 0.0;
            for (uint32_t k = 0; k < gd; k++) s += (double)x[(uint64_t)t * in_all + g * gd + k] * wref[((uint64_t)g * rank + r) * gd + k];
            ref[(uint64_t)t * out_all + g * rank + r] = test_v41_bf16r((float)s);
        }
        char name[64]; snprintf(name, sizeof name, "%s grouped n=%u", wt_name(wt), n);
        test_v41_cmp(name, got, ref, (uint64_t)n * out_all, 6e-3f, 1e-3f);
        /* 分组转置乘: gheads[n][G·gd] = glow[n][G·rank]·W_g */
        {
            float *gh = malloc((size_t)n * in_all * 4), *gref = malloc((size_t)n * in_all * 4);
            ds4_gpu_tensor *tgh = test_v41_tensor(NULL, (uint64_t)n * in_all * 4);
            TEST_ASSERT(ds4_gpu_bwd_grouped_matmul_t_tensor(tgh, test_v41_model_map(), test_v41_model_size(), wt_ggt(wt), off, G, gd, rank, to, n, 0) != 0);
            TEST_ASSERT(test_v41_read(tgh, gh, (uint64_t)n * in_all * 4));
            for (uint32_t t = 0; t < n; t++) for (uint32_t g = 0; g < G; g++) for (uint32_t k = 0; k < gd; k++) {
                double s = 0.0;
                for (uint32_t r = 0; r < rank; r++) s += (double)got[(uint64_t)t * out_all + g * rank + r] * wref[((uint64_t)g * rank + r) * gd + k];
                gref[(uint64_t)t * in_all + g * gd + k] = (float)s;
            }
            snprintf(name, sizeof name, "%s grouped_t n=%u", wt_name(wt), n);
            test_v41_cmp(name, gh, gref, (uint64_t)n * in_all, 2e-4f, 1e-3f);
            ds4_gpu_tensor_free(tgh); free(gh); free(gref);
        }
        ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(to); free(x); free(ref); free(got);
    }
    free(wref);
}

/* f32 GEMM 经公开入口验: amp_apply 走 NT + NN 两种转置, bwd_amp(t_metal_v41_bwd.c)走另两种。
 * amp_apply: y[n][D] += x[n][D]·(B·A), A/B [Kr][D] */
static void test_sgemm(void) {
    float *ref, *got;
    const uint32_t n = 13, D = 96, Kr = 24;
    ref = malloc((size_t)n * D * 4); got = malloc((size_t)n * D * 4);
    float *x = malloc((size_t)n * D * 4), *Am = malloc((size_t)Kr * D * 4), *Bm = malloc((size_t)Kr * D * 4), *y = malloc((size_t)n * D * 4);
    for (uint32_t i = 0; i < n * D; i++) { x[i] = test_v41_randf(); y[i] = test_v41_randf(); }
    for (uint32_t i = 0; i < Kr * D; i++) { Am[i] = test_v41_randf(); Bm[i] = test_v41_randf(); }
    ds4_gpu_tensor *tx = test_v41_tensor(x, (uint64_t)n * D * 4), *tA = test_v41_tensor(Am, (uint64_t)Kr * D * 4), *tB = test_v41_tensor(Bm, (uint64_t)Kr * D * 4);
    ds4_gpu_tensor *ty = test_v41_tensor(y, (uint64_t)n * D * 4), *tT = test_v41_tensor(NULL, (uint64_t)n * Kr * 4);
    TEST_ASSERT(ds4_gpu_v41_amp_apply_tensor(ty, tx, tA, tB, tT, n, D, Kr) != 0);
    TEST_ASSERT(test_v41_read(ty, got, (uint64_t)n * D * 4));
    for (uint32_t t = 0; t < n; t++) for (uint32_t d = 0; d < D; d++) {
        double s = y[(uint64_t)t * D + d];
        for (uint32_t k = 0; k < Kr; k++) { double tk = 0.0; for (uint32_t e = 0; e < D; e++) tk += (double)x[(uint64_t)t * D + e] * Bm[(uint64_t)k * D + e]; s += tk * Am[(uint64_t)k * D + d]; }
        ref[(uint64_t)t * D + d] = (float)s;
    }
    test_v41_cmp("amp_apply y+=x(BA)", got, ref, (uint64_t)n * D, 2e-4f, 1e-3f);
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tA); ds4_gpu_tensor_free(tB); ds4_gpu_tensor_free(ty); ds4_gpu_tensor_free(tT);
    free(x); free(Am); free(Bm); free(y); free(ref); free(got);
}

static void test_embed_colnorm(void) {
    const uint32_t V = 50, D = 512;
    const uint32_t wts[2] = { V41_WT_FP4X32, V41_WT_Q4K };
    for (int c = 0; c < 2; c++) {
        const uint32_t wt = wts[c];
        const uint64_t wb = test_v41_wbytes(wt, V, D);
        TEST_ASSERT(test_v41_model_begin(wb + 4096));
        const uint64_t off = test_v41_model_alloc(wb);
        float *wref = malloc((size_t)V * D * 4);
        test_v41_make_weights(wt, V, D, test_v41_model_ptr(off), wref);
        TEST_ASSERT(test_v41_map());
        const int32_t tok[3] = { 7, 49, 0 };
        ds4_gpu_tensor *tt = test_v41_tensor(tok, 12), *to = test_v41_tensor(NULL, 3u * D * 4);
        if (wt == V41_WT_FP4X32) TEST_ASSERT(ds4_gpu_v41_embed_fp4x32_tensor(to, tt, test_v41_model_map(), test_v41_model_size(), off, V, 3, D) != 0);
        else TEST_ASSERT(ds4_gpu_v41_embed_q4k_tensor(to, tt, test_v41_model_map(), test_v41_model_size(), off, V, 3, D) != 0);
        float *got = malloc(3u * D * 4), *ref = malloc(3u * D * 4);
        TEST_ASSERT(test_v41_read(to, got, 3u * D * 4));
        for (int t = 0; t < 3; t++) for (uint32_t d = 0; d < D; d++) ref[t * D + d] = wt == V41_WT_FP4X32 ? test_v41_bf16r(wref[(uint64_t)tok[t] * D + d]) : wref[(uint64_t)tok[t] * D + d];
        test_v41_cmp(wt == V41_WT_FP4X32 ? "embed fp4x32" : "embed q4_K", got, ref, 3u * D, 1e-6f, 1e-3f);
        ds4_gpu_tensor *tc = test_v41_tensor(NULL, (uint64_t)D * 4);
        if (wt == V41_WT_FP4X32) TEST_ASSERT(ds4_gpu_v41_head_colnorm_tensor(tc, test_v41_model_map(), test_v41_model_size(), off, V, D) != 0);
        else TEST_ASSERT(ds4_gpu_v41_head_colnorm_q4k_tensor(tc, test_v41_model_map(), test_v41_model_size(), off, V, D) != 0);
        TEST_ASSERT(test_v41_read(tc, got, (uint64_t)D * 4));
        for (uint32_t d = 0; d < D; d++) { double s = 0.0; for (uint32_t v = 0; v < V; v++) s += (double)wref[(uint64_t)v * D + d] * wref[(uint64_t)v * D + d]; ref[d] = (float)s; }
        test_v41_cmp(wt == V41_WT_FP4X32 ? "colnorm fp4x32" : "colnorm q4_K", got, ref, D, 1e-5f, 1e-3f);
        ds4_gpu_tensor_free(tt); ds4_gpu_tensor_free(to); ds4_gpu_tensor_free(tc); free(got); free(ref); free(wref);
    }
}

void test_metal_v41_dense(void) {
    test_v41_seed(0x41);
    for (uint32_t wt = 0; wt < 5; wt++) test_dense_type(wt);
    test_grouped_type(V41_WT_FP4X32); test_grouped_type(V41_WT_Q4K); test_grouped_type(V41_WT_FP8BLK);
    test_sgemm();
    test_embed_colnorm();
}
#else
typedef int ds4_t_metal_v41_dense_nonempty_tu;
#endif
