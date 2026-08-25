/* core_convert.c — e4m3/e2m1/hadamard/fp4/indexer_qat 标量转换 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
/* E4M3/E2M1 标量基元收敛到全仓唯一实现(重构阶段: src/common 替换)。
 * ds4_e4m3fn_value/round 与 ds4_e2m1fn_value/round 是原 dsv4_*_cpu 的逐式
 * 转录(见 src/common/ds4_fp8.h 数值契约), 字节语义不变。 */
#include "src/common/ds4_fp8.h"
/* =========================================================================
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These functions are the CPU reference math used by the C backend and by
 * Metal diagnostics.  They implement only the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q8_0, Q2_K, IQ2_XXS, and Q8_K activation
 * blocks used for expert dot products.
 */

void f16_round_inplace_cpu(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) x[i] = f16_to_f32(f32_to_f16(x[i]));
}

/* DeepSeek V4 stores the non-RoPE part of compressed KV through an E4M3-style
 * round trip.  Keeping this in the CPU reference makes cache values comparable
 * to the Metal graph's compressed-cache behavior. */
void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }

        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = ds4_e4m3fn_round(v) * scale;
        }
    }
}

static void dsv4_hadamard128_inplace_cpu(float *x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1) {
        for (uint32_t base = 0; base < 128; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.08838834764831845f;
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}

static void dsv4_fp4_act_quantize_row_inplace_cpu(float *x, uint32_t n) {
    if ((n % 32u) != 0) ds4_die("DSV4 FP4 activation quantization requires 32-aligned rows");
    for (uint32_t off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }

        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (uint32_t i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = ds4_e2m1fn_round(v) * scale;
        }
    }
}

/* The official DeepSeek V4 graph rotates indexer activations with a 128-wide
 * Hadamard transform and immediately runs the FP4 activation-simulation
 * round trip. This applies to both indexer Q and the indexer compressor KV;
 * without it, the top-k compressed-row selection is not the model's graph. */
void dsv4_indexer_qat_row_inplace_cpu(float *x, uint32_t head_dim) {
    if (head_dim != 128) ds4_die("DSV4 indexer QAT expects 128-wide indexer rows");
    dsv4_hadamard128_inplace_cpu(x);
    dsv4_fp4_act_quantize_row_inplace_cpu(x, head_dim);
}

void dsv4_indexer_qat_rows_inplace_cpu(float *x, uint32_t rows, uint32_t head_dim) {
    for (uint32_t r = 0; r < rows; r++) {
        dsv4_indexer_qat_row_inplace_cpu(x + (uint64_t)r * head_dim, head_dim);
    }
}

/* Quantize a float activation into Q8_K blocks so GGUF Q2_K/IQ2_XXS expert
 * kernels can reuse the same activation for many expert rows. */
