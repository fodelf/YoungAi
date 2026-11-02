/* Bounded model-free Q4_K benchmark. Run the same source on both revisions. */
#include "../test_internal.h"
#include "../../ds4_gpu_v41.h"
#include "../../src/metal/metal_v41_args.h"
#include <time.h>

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
int main(void) {
    const uint32_t K = 5120, O = 32768, REPS = 64;
    const uint64_t bytes = test_v41_wbytes(V41_WT_Q4K, O, K);
    test_v41_seed(0x6161);
    if (!ds4_gpu_init() || !test_v41_model_begin(bytes + 4096)) return 1;
    const uint64_t off = test_v41_model_alloc(bytes);
    float *ref = malloc((size_t)O * K * 4), *x = malloc(8u * K * 4);
    if (!ref || !x) return 1;
    test_v41_make_weights(V41_WT_Q4K, O, K, test_v41_model_ptr(off), ref);
    free(ref); test_v41_fill_bf16(x, 8u * K, 1.f);
    if (!test_v41_map()) return 1;
    ds4_gpu_tensor *tx = test_v41_tensor(x, 8u * K * 4), *out = test_v41_tensor(NULL, 8u * O * 4);
    free(x);
    for (uint32_t n = 1; n <= 8; n++) {
        /* Warm the selected pipeline before five timed command batches. */
        if (!ds4_gpu_begin_commands()) return 1;
        for (uint32_t i = 0; i < REPS; i++)
            if (!ds4_gpu_v41_matmul_q4k_tensor(out, test_v41_model_map(), test_v41_model_size(), off, K, O, tx, n, 1)) return 1;
        if (!ds4_gpu_end_commands()) return 1;
        for (int run = 0; run < 5; run++) {
            const double begin = seconds();
            if (!ds4_gpu_begin_commands()) return 1;
            for (uint32_t i = 0; i < REPS; i++)
                if (!ds4_gpu_v41_matmul_q4k_tensor(out, test_v41_model_map(), test_v41_model_size(), off, K, O, tx, n, 1)) return 1;
            if (!ds4_gpu_end_commands()) return 1;
            printf("{\"rows\":%u,\"run\":%d,\"weight_bytes\":%llu,\"ms\":%.6f}\n", n, run,
                   (unsigned long long)bytes, (seconds() - begin) * 1000. / REPS);
            fflush(stdout);
        }
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(out); ds4_gpu_cleanup();
    return 0;
}
