#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include "../ds4_spatial.h"
#include "../ds4_css.h"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = quality,
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

/* wave-78 gate: a non-%32 batch (e.g. copy-spec verify batch of 49) must be
 * bit-exact through the NAX-padding path. The activation/output buffers are
 * sized to the padded row count (64) so the matmul's capacity guard engages
 * n_tok_nax=64; rows [0,49) must match the CPU reference, pad rows are ignored. */
static void test_metal_q8_0_prefill_matmul_unaligned(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 49;          /* logical batch (not a multiple of 32) */
    const uint32_t n_tok_alloc = 64;    /* padded capacity the matmul rounds up to */
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok_alloc * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok_alloc * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    /* Fill all padded rows so the pad rows hold deterministic finite values. */
    for (uint32_t t = 0; t < n_tok_alloc; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok_alloc * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    /* call with the unaligned logical n_tok; the matmul pads to 64 internally */
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {   /* only the real rows must be correct */
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

/* TP Phase 1 correctness gate: the row-parallel Q8_0 matvec split must reconstruct
 * the full matvec after the (host) all-reduce. Splits the in-dim into two halves,
 * computes each peer's partial out via ds4_gpu_matmul_q8_0_rowslice_tensor (the new
 * primitive: ne00=half blocks accumulated, nb01=full row stride for addressing),
 * sums them, and checks against the CPU reference. The two-half sum differs from a
 * single-machine reduction only in float associativity (~1e-6) — well inside the
 * Q8_0 tolerance. Proves the shared-FFN TP down projection before any dual-host run. */
static void test_metal_q8_0_rowslice_tp(void) {
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

static void test_metal_go1b_routed_moe(void) {
    /* Exercise the over-budget OFFLOAD dispatch (CPU gather -> resident scratch ->
     * mm_id). pread auto-falls back to memcpy with no model fd; force it off so the
     * synthetic host-buffer model is gathered by memcpy. */
    setenv("DS4_METAL_EXPERT_OFFLOAD", "1", 1);
    setenv("DS4_METAL_EXPERT_PREAD", "0", 1);

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
static void test_metal_go2b_routed_moe(void) {
    setenv("DS4_METAL_EXPERT_OFFLOAD", "1", 1);
    setenv("DS4_METAL_EXPERT_PREAD", "0", 1);
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

static void test_metal_kernel_group(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_q8_0_prefill_matmul_unaligned();
    test_metal_q8_0_rowslice_tp();
    test_metal_go1b_routed_moe();
    test_metal_go2b_routed_moe();
    test_metal_corr_apply();
}

static void test_metal_short_prefill_ratio4(void) {
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

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors(void) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (test_logprob_vector_case_disabled(&vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
    }
    ds4_engine_close(engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "ds4-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/local-golden.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    ds4_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    ds4_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

#endif

/* ---- 前端域多模态插件 (物理方位 / CSS 理解 / enricher 链) ---- */

/* mm-ui selftest 场景的手写草图副本: 数字取整, "Continue" 精确居中。
 * 格式契约与 tools/mm_ui.swift 输出一致 -- 这里断言的每个空间/CSS 事实
 * 都能手算复核。 */
static const char *TEST_UI_SKETCH =
    "[img 1280x800]\n"
    "[palette #f5f6f8 84% #101828 8%]\n"
    "[rect 0,65 1280x735 #f5f6f8]\n"
    "[rect 0,0 1280x64 #101828]\n"
    "[rect 480,280 320x240 #ffffff]\n"
    "[rect 512,436 256x44 #3b82f6]\n"
    "[text 40,22 146x22 #ffffff on #101828 \"Acme Console\"]\n"
    "[text 512,318 74x28 #101828 on #ffffff \"Sign in\"]\n"
    "[text 596,446 88x24 #ffffff on #3b82f6 \"Continue\"]\n";

static void test_spatial_annotate_facts(void) {
    char *sec = ds4_spatial_annotate(TEST_UI_SKETCH);
    TEST_ASSERT(sec != NULL);
    TEST_ASSERT(strstr(sec, "t2\"Sign in\"") != NULL);           /* 标签系统 */
    TEST_ASSERT(strstr(sec, "r1 page") != NULL);                 /* 九宫格方位 */
    TEST_ASSERT(strstr(sec, "r2 top") != NULL);
    TEST_ASSERT(strstr(sec, "r3 center") != NULL);
    TEST_ASSERT(strstr(sec, "[in r3(480,280): t2 r4]") != NULL); /* 包含树 */
    TEST_ASSERT(strstr(sec, "[stack r3 column gap=90px: t2 r4]") != NULL);
    TEST_ASSERT(strstr(sec, "[align left: t2 r4 (in r3)]") != NULL);
    TEST_ASSERT(strstr(sec, "[align centered-x in r3: r4]") != NULL);
    TEST_ASSERT(strstr(sec, "[align centered-x in r4: t3]") != NULL);
    free(sec);
    /* 非草图输入诚实缺席 */
    TEST_ASSERT(ds4_spatial_annotate("hello, not a sketch") == NULL);
}

static void test_css_annotate_facts(void) {
    char *sec = ds4_css_annotate(TEST_UI_SKETCH);
    TEST_ASSERT(sec != NULL);
    TEST_ASSERT(strstr(sec, "[css page: background:#f5f6f8]") != NULL);
    TEST_ASSERT(strstr(sec, "[css r3: background:#ffffff; "
                            "padding:38px 32px 40px 32px; display:flex; "
                            "flex-direction:column; gap:90px; "
                            "align-items:flex-start]") != NULL);
    TEST_ASSERT(strstr(sec, "[css r4: background:#3b82f6; "
                            "padding:10px 84px 10px 84px]") != NULL);
    TEST_ASSERT(strstr(sec, "[css r2: background:#101828; "
                            "padding:22px 1094px 20px 40px]") != NULL);
    free(sec);
    TEST_ASSERT(ds4_css_annotate("plain text") == NULL);
}

static void test_mm_stub_tokenize(void *ctx, const char *text, int **toks, int *n) {
    (void)ctx;
    const size_t len = strlen(text);
    *toks = malloc(len ? len * sizeof(int) : sizeof(int));
    *n = 0;
    if (!*toks) return;
    for (size_t i = 0; i < len; i++) (*toks)[i] = (int)(unsigned char)text[i];
    *n = (int)len;
}

static char *test_mm_spatial_adapter(void *ctx, const char *modality,
                                     const char *text) {
    (void)ctx;
    (void)modality;
    return ds4_spatial_annotate(text);
}

static char *test_mm_css_adapter(void *ctx, const char *modality,
                                 const char *text) {
    (void)ctx;
    (void)modality;
    return ds4_css_annotate(text);
}

static void test_mm_enrichers_compose_both_forms(void) {
    ds4_mm *mm = ds4_mm_create(test_mm_stub_tokenize, NULL);
    TEST_ASSERT(mm != NULL);
    TEST_ASSERT(ds4_mm_register_command(mm, "image", "/bin/cat") == 0);
    TEST_ASSERT(ds4_mm_register_enricher(mm, "image", test_mm_spatial_adapter,
                                         NULL) == 0);
    TEST_ASSERT(ds4_mm_register_enricher(mm, "image", test_mm_css_adapter,
                                         NULL) == 0);
    char *text = NULL;
    TEST_ASSERT(ds4_mm_encode_as_text(mm, "image/png",
                                      (const uint8_t *)TEST_UI_SKETCH,
                                      strlen(TEST_UI_SKETCH), &text) == 0);
    TEST_ASSERT(text != NULL);
    const char *sketch_tail = strstr(text, "\"Continue\"]");
    const char *spatial = strstr(text, "[in r3");
    const char *css = strstr(text, "[css r3");
    TEST_ASSERT(sketch_tail && spatial && css);
    TEST_ASSERT(sketch_tail < spatial && spatial < css);  /* 节序 = 注册序 */
    const size_t text_len = strlen(text);
    /* token 形必须携带同一份增强文本 (字节 stub tokenizer 逐字节出 token) */
    int *toks = NULL;
    int n_toks = 0;
    TEST_ASSERT(ds4_mm_encode(mm, "image/png",
                              (const uint8_t *)TEST_UI_SKETCH,
                              strlen(TEST_UI_SKETCH), &toks, &n_toks) == 0);
    TEST_ASSERT((size_t)n_toks == text_len);
    free(toks);
    free(text);
    /* text 模态没挂 enricher: identity 原样 */
    TEST_ASSERT(ds4_mm_encode_as_text(mm, "text", (const uint8_t *)"hi", 2,
                                      &text) == 0);
    TEST_ASSERT(text && !strcmp(text, "hi"));
    free(text);
    ds4_mm_free(mm);
}

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
    test_spatial_annotate_facts();
    test_css_annotate_facts();
    test_mm_enrichers_compose_both_forms();
}

/* TP Stage 2: in-process loopback check of the all-reduce transport (frame
 * format + sum correctness + no deadlock). No model, no network. */
static void test_tp_allreduce(void) {
    TEST_ASSERT(ds4_dist_tp_selftest() == 0);
}

/* ---- Model-free unit tests for the repeat penalty core ----
 *
 * Exercises ds4_repeat_penalize_tokens -- the raw-array entry that shares
 * repeat_penalize_core with the session sampler -- on synthetic committed
 * streams: no model, no engine, no GPU. The DS4_REPEAT_FREQ /
 * DS4_REPEAT_WINDOW caches inside ds4.c are process-wide
 * statics armed on first use, so the
 * suite pins each config via the test-only reset hook (setenv BEFORE the
 * first penalize call of that config) and restores the developer's env +
 * re-resets afterwards, so greedy-sensitive model suites later in the same
 * --all run re-arm from their own environment. */
extern void ds4_test_reset_penalty_env_cache(void);

/* Token ids in these tests stay far below this: the penalty core only ever
 * writes logits at committed-token indices, so a small row suffices. */
#define TEST_PEN_NLOGITS 256

static char *test_penalty_env_save(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (copy) memcpy(copy, value, len + 1);
    return copy;
}

static void test_penalty_env_restore(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

static void test_penalize_row(float *logits, const int *toks, uint32_t n,
                              uint32_t gen_start) {
    for (int i = 0; i < TEST_PEN_NLOGITS; i++) logits[i] = 0.0f;
    ds4_repeat_penalize_tokens(logits, toks, n, gen_start);
}

static int test_pen_argmax(const float *logits) {
    int best = 0;
    for (int i = 1; i < TEST_PEN_NLOGITS; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* n copies of {A(8 tokens 10..17), 42, distinct gap} + a final A: the current
 * 8-suffix A occurred n times before in the stream, every time continued by
 * 42, while the distinct gap tokens kill every exact period P<=64, isolating
 * the 8-gram self-copy detector. Returns the stream length (10n+8). */
static uint32_t test_pen_build_selfcopy(int *sc, int n) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 8; j++) sc[m++] = 10 + j;
        sc[m++] = 42;
        sc[m++] = 90 + i;
    }
    for (int j = 0; j < 8; j++) sc[m++] = 10 + j;
    return (uint32_t)m;
}

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
    {"--tp-allreduce", "tp-allreduce", "tensor-parallel all-reduce transport loopback", test_tp_allreduce},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local top-k golden-vector fixture.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
