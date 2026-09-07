#include "ds4_gpu.h"
#include "src/common/ds4_float.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        /* 2s: JIT 后复跑的 top-k 墙钟红线(首跑 PTX JIT 不算数, 见 CLAUDE.md)。 */
        const double max_seconds = 2.0;
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    /* 压缩缓存行格式(ds4_gpu_core.h): [448 维 f16][64 维 f32], 喂核之前先按缓存格式打包 */
    uint8_t *comp_rows = (uint8_t *)calloc((size_t)n_comp, DS4_GPU_COMP_ROW_BYTES);
    if (!comp_rows) return 1;
    for (uint32_t c = 0; c < n_comp; c++) {
        uint8_t *row = comp_rows + (size_t)c * DS4_GPU_COMP_ROW_BYTES;
        const float *src = comp_host + (uint64_t)c * head_dim;
        for (uint32_t d = 0; d < DS4_GPU_COMP_ROW_NOPE; d++) ((uint16_t *)(void *)row)[d] = ds4_f64_to_f16((double)src[d]);
        memcpy(row + DS4_GPU_COMP_ROW_NOPE * 2u, src + DS4_GPU_COMP_ROW_NOPE, DS4_GPU_COMP_ROW_ROT * sizeof(float));
    }
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)n_comp * DS4_GPU_COMP_ROW_BYTES);
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_rows, (uint64_t)n_comp * DS4_GPU_COMP_ROW_BYTES) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(comp_rows);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

/* 09-07: 大 n_comp 的 top-k 多 block 版(cuda_indexer_kernels_6, ≤8 token 走)必须与单 block 版(kernels_5, >8 token 走)
 * 选集与顺序逐字同 —— 投机 verify 批(多 block)与纯解码(同一路)之外, prefill 大批走单 block, 两路选集不同就是稀疏区分叉。
 * 同一份分数: 前 4 个 token 按 4 token 发一次(多 block), 再按 9 token 发一次(单 block), 逐 token 逐位比。
 * 分数带大量并列(取模 997)专门考并列的索引序规则; 1M 上下文 n_comp = 262144。 */
static int check_topk_multi_parity(void) {
    const uint32_t n_comp = 262144u, top_k = 512u, n_multi = 4u, n_single = 9u;
    const uint64_t rows = (uint64_t)n_single * n_comp;
    float *scores_host = (float *)malloc((size_t)rows * sizeof(float));
    uint32_t *sel_multi = (uint32_t *)malloc((size_t)n_multi * top_k * sizeof(uint32_t));
    uint32_t *sel_single = (uint32_t *)malloc((size_t)n_single * top_k * sizeof(uint32_t));
    if (!scores_host || !sel_multi || !sel_single) return 1;
    uint32_t h = 0x9e3779b9u;
    for (uint32_t t = 0; t < n_single; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            h ^= h << 13; h ^= h >> 17; h ^= h << 5;   /* xorshift, 确定性 */
            const float v = (float)(h % 997u) - 500.0f + (float)(t % n_multi) * 0.25f;
            scores_host[(uint64_t)(t % n_multi) * n_comp + i] = v;   /* token t 与 t%4 同行: 单 block 跑的 4..8 复用 0..3 */
        }
    }
    for (uint32_t t = n_multi; t < n_single; t++)
        memcpy(scores_host + (uint64_t)t * n_comp, scores_host + (uint64_t)(t % n_multi) * n_comp, (size_t)n_comp * sizeof(float));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(rows * sizeof(float));
    ds4_gpu_tensor *sel_m = ds4_gpu_tensor_alloc((uint64_t)n_multi * top_k * sizeof(uint32_t));
    ds4_gpu_tensor *sel_s = ds4_gpu_tensor_alloc((uint64_t)n_single * top_k * sizeof(uint32_t));
    int rc = 1;
    if (scores && sel_m && sel_s &&
        ds4_gpu_tensor_write(scores, 0, scores_host, rows * sizeof(float)) &&
        ds4_gpu_indexer_topk_tensor(sel_m, scores, n_comp, n_multi, top_k) &&
        ds4_gpu_indexer_topk_tensor(sel_s, scores, n_comp, n_single, top_k) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(sel_m, 0, sel_multi, (uint64_t)n_multi * top_k * sizeof(uint32_t)) &&
        ds4_gpu_tensor_read(sel_s, 0, sel_single, (uint64_t)n_single * top_k * sizeof(uint32_t))) {
        rc = 0;
        for (uint32_t t = 0; t < n_multi && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t a = sel_multi[(uint64_t)t * top_k + i], b = sel_single[(uint64_t)t * top_k + i];
                if (a != b) {
                    fprintf(stderr, "top-k multi/single mismatch token=%u rank=%u multi=%u single=%u\n", t, i, a, b);
                    rc = 1;
                    break;
                }
            }
        }
        if (rc == 0) fprintf(stderr, "cuda-regression: top-k multi-block parity n_comp=%u OK\n", n_comp);
    }
    ds4_gpu_tensor_free(sel_s);
    ds4_gpu_tensor_free(sel_m);
    ds4_gpu_tensor_free(scores);
    free(sel_single);
    free(sel_multi);
    free(scores_host);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_topk_multi_parity() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
