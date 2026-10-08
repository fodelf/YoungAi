/* t_metal_v41_util.c — V4.1 Metal 原语数值回归的公共件(2026-10-08): 合成"模型映射"、各类型合成权重 + 金标解码、张量搬运、比对。
 * 金标 = src/common 的标量解码(ds4_deq_bytes / ds4_fp8.h), 与引擎 CUDA 路共用的同一份; 比对口径 = 最大绝对误差 ≤ tol × 参考 RMS。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_quantfmt.h"
#include "../src/common/ds4_fp8.h"
#include "../src/metal/metal_v41_args.h"

static uint64_t g_v41_rng = 0x9E3779B97F4A7C15ull;
void test_v41_seed(uint64_t s) { g_v41_rng = s ? s : 1; }
uint32_t test_v41_rand_u32(void) {
    uint64_t z = (g_v41_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
    return (uint32_t)(z >> 32);
}
float test_v41_randf(void) { return (float)test_v41_rand_u32() / 4294967296.0f * 2.0f - 1.0f; }   /* [-1, 1) */
float test_v41_bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u); u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

/* 合成模型映射: 一块页对齐内存, 子分配按 32 B 对齐(与 GGUF 数据区同), 用完 test_v41_map 一次注册成模型 */
static uint8_t *g_v41_model; static uint64_t g_v41_model_cap, g_v41_model_used;
int test_v41_model_begin(uint64_t bytes) {
    free(g_v41_model); g_v41_model = NULL;
    g_v41_model_cap = test_round_up_u64(bytes + 64u, (uint64_t)getpagesize()); g_v41_model_used = 0;
    void *p = NULL;
    if (posix_memalign(&p, (size_t)getpagesize(), (size_t)g_v41_model_cap) != 0) return 0;
    memset(p, 0, (size_t)g_v41_model_cap);
    g_v41_model = p;
    return 1;
}
uint64_t test_v41_model_alloc(uint64_t bytes) {
    const uint64_t off = test_round_up_u64(g_v41_model_used, 32u);
    if (off + bytes > g_v41_model_cap) { fprintf(stderr, "test_v41: 合成模型容量不够(%llu + %llu > %llu)\n", (unsigned long long)off, (unsigned long long)bytes, (unsigned long long)g_v41_model_cap); return UINT64_MAX; }
    g_v41_model_used = off + bytes;
    return off;
}
uint8_t *test_v41_model_ptr(uint64_t off) { return g_v41_model + off; }
const void *test_v41_model_map(void) { return g_v41_model; }
uint64_t test_v41_model_size(void) { return g_v41_model_cap; }
int test_v41_map(void) { return ds4_gpu_set_model_map(g_v41_model, g_v41_model_cap) != 0; }

/* 一个 [rows][cols] 矩阵: 合成字节写进 dst, 金标 f32 写进 ref(rows*cols)。类型号 V41_WT_*(metal_v41_args.h)。 */
uint64_t test_v41_wbytes(uint32_t wt, uint64_t rows, uint64_t cols) {
    switch (wt) {
        case V41_WT_FP4X32: return rows * (cols / 32u) * 17u;
        case V41_WT_Q4K: return rows * (cols / 256u) * 144u;
        case V41_WT_BF16: return rows * cols * 2u;
        case V41_WT_F32: return rows * cols * 4u;
        default: return rows * cols + ((rows + 31u) / 32u) * ((cols + 31u) / 32u);
    }
}
void test_v41_make_weights(uint32_t wt, uint64_t rows, uint64_t cols, uint8_t *dst, float *ref) {
    const uint64_t n = rows * cols;
    if (wt == V41_WT_FP4X32) {
        float *tmp = malloc(n * 4);
        for (uint64_t i = 0; i < n; i++) tmp[i] = test_v41_randf() * (1.0f + (float)(i % 7u));
        ds4_quant_fp4x32(tmp, n / 32u, dst);
        ds4_deq_fp4x32(dst, n / 32u, ref);
        free(tmp);
    } else if (wt == V41_WT_Q4K) {
        for (uint64_t b = 0; b < n / 256u; b++) {
            uint8_t *blk = dst + b * 144u;
            const uint16_t d = test_float_to_f16(0.01f + 0.02f * (float)(test_v41_rand_u32() % 17u)), dm = test_float_to_f16(0.005f * (float)(test_v41_rand_u32() % 9u));
            memcpy(blk, &d, 2); memcpy(blk + 2, &dm, 2);
            for (int i = 4; i < 144; i++) blk[i] = (uint8_t)test_v41_rand_u32();
        }
        ds4_deq_q4_K(dst, n / 256u, ref);
    } else if (wt == V41_WT_BF16) {
        for (uint64_t i = 0; i < n; i++) { const float v = test_v41_bf16r(test_v41_randf()); uint32_t u; memcpy(&u, &v, 4); const uint16_t h = (uint16_t)(u >> 16); memcpy(dst + i * 2, &h, 2); ref[i] = v; }
    } else if (wt == V41_WT_F32) {
        for (uint64_t i = 0; i < n; i++) { ref[i] = test_v41_randf(); memcpy(dst + i * 4, &ref[i], 4); }
    } else {
        const uint64_t sbc = (cols + 31u) / 32u, sbr = (rows + 31u) / 32u;
        uint8_t *sc = dst + n;
        for (uint64_t i = 0; i < sbr * sbc; i++) sc[i] = (uint8_t)(120u + test_v41_rand_u32() % 8u);
        for (uint64_t i = 0; i < n; i++) {
            uint8_t b = (uint8_t)test_v41_rand_u32();
            if ((b & 0x7f) == 0x7f) b &= 0x7e;   /* 避开 NaN 槽 */
            dst[i] = b;
            ref[i] = ds4_e4m3fn_to_f32(b) * ds4_e8m0_to_f32(sc[(i / cols / 32u) * sbc + (i % cols) / 32u]);
        }
    }
}
/* 随机 bf16 格点激活(引擎里进 GEMV 的激活本来就在格点上) */
void test_v41_fill_bf16(float *x, uint64_t n, float scale) { for (uint64_t i = 0; i < n; i++) x[i] = test_v41_bf16r(test_v41_randf() * scale); }
ds4_gpu_tensor *test_v41_tensor(const void *src, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (t && src) (void)ds4_gpu_tensor_write(t, 0, src, bytes);
    else if (t) { void *p = ds4_gpu_tensor_contents(t); memset(p, 0, (size_t)bytes); }
    return t;
}
int test_v41_read(const ds4_gpu_tensor *t, void *dst, uint64_t bytes) {
    if (!ds4_gpu_synchronize()) return 0;
    return ds4_gpu_tensor_read(t, 0, dst, bytes);
}
/* 比对: 最大绝对误差 ≤ tol × max(参考 RMS, floor); 打一行读数(用户要看原始输出) */
int test_v41_cmp(const char *name, const float *got, const float *ref, uint64_t n, float tol, float floor_) {
    double ss = 0.0; float max_abs = 0.0f; uint64_t at = 0; int bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!(got[i] == got[i]) || !(ref[i] == ref[i])) { bad = 1; at = i; break; }
        ss += (double)ref[i] * ref[i];
        const float e = fabsf(got[i] - ref[i]);
        if (e > max_abs) { max_abs = e; at = i; }
    }
    const float rms = (float)sqrt(ss / (double)(n ? n : 1)), lim = tol * (rms > floor_ ? rms : floor_);
    const int ok = !bad && max_abs <= lim;
    fprintf(stderr, "ds4: [v41-test] %-28s n=%llu ref_rms=%.5g max_abs=%.3g (限 %.3g) @%llu got=%.6g ref=%.6g %s\n", name, (unsigned long long)n, rms, max_abs, lim,
            (unsigned long long)at, n ? got[at] : 0.f, n ? ref[at] : 0.f, ok ? "ok" : "★FAIL★");
    TEST_ASSERT(ok);
    return ok;
}
/* CPU 参考: out[M][N] = x[M][K]·Wᵀ(W [N][K]) */
void test_v41_ref_matmul(const float *x, const float *w, float *out, uint32_t M, uint32_t N, uint32_t K, int round_out) {
    for (uint32_t m = 0; m < M; m++) for (uint32_t n = 0; n < N; n++) {
        double s = 0.0;
        for (uint32_t k = 0; k < K; k++) s += (double)x[(uint64_t)m * K + k] * w[(uint64_t)n * K + k];
        out[(uint64_t)m * N + n] = round_out ? test_v41_bf16r((float)s) : (float)s;
    }
}
typedef int ds4_t_metal_v41_util_nonempty_tu;
#else
typedef int ds4_t_metal_v41_util_nonempty_tu;
#endif
