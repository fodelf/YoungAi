/* Model-free regressions. Keep this file usable on the parent revision: the
 * same tests must expose the old upload race. */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)
/* Uploads inside a command batch must not overwrite bytes that an earlier GPU
 * operation still needs. V4.1 reuses its compressed-key position buffer this way. */
static void test_ordered_upload(void) {
    for (int batched = 0; batched <= 1; batched++) {
        const float first[4] = { 1, 2, 3, 4 }, second[4] = { 5, 6, 7, 8 }, third[4] = { 9, 10, 11, 12 };
        ds4_gpu_tensor *src = test_v41_tensor(first, sizeof first), *dst = test_v41_tensor(NULL, 3 * sizeof first);
        if (batched) TEST_ASSERT(ds4_gpu_begin_commands());
        TEST_ASSERT(ds4_gpu_tensor_copy(dst, 0, src, 0, sizeof first));
        TEST_ASSERT(ds4_gpu_tensor_write_zerocopy(src, 0, second, sizeof second));
        TEST_ASSERT(ds4_gpu_tensor_copy(dst, sizeof first, src, 0, sizeof first));
        TEST_ASSERT(ds4_gpu_tensor_write(src, 0, third, sizeof third));
        TEST_ASSERT(ds4_gpu_tensor_copy(dst, 2 * sizeof first, src, 0, sizeof first));
        if (batched) TEST_ASSERT(ds4_gpu_end_commands());
        else TEST_ASSERT(ds4_gpu_synchronize());
        float got[12], expected[12];
        memcpy(expected, first, sizeof first); memcpy(expected + 4, second, sizeof second); memcpy(expected + 8, third, sizeof third);
        TEST_ASSERT(test_v41_read(dst, got, sizeof got));
        test_v41_cmp(batched ? "ordered batched uploads" : "ordered async uploads", got, expected, 12, 0.f, 0.f);
        ds4_gpu_tensor_free(src); ds4_gpu_tensor_free(dst);
    }
}

/* Actual plain-decode hazard: at odd position p, ratio=2 sources upload p-1
 * and the ratio=1 source uploads p into the same posg before the batch runs. */
static void test_decode_positions(void) {
    const uint32_t D = 512;
    float input[512], got[2][512], expected[2][512];
    for (uint32_t i = 0; i < D; i++) input[i] = (float)(i % 17) * .03125f;
    ds4_gpu_tensor *pos = test_v41_tensor(NULL, 4);
    ds4_gpu_tensor *x[2] = {test_v41_tensor(input, sizeof input), test_v41_tensor(input, sizeof input)};
    const int32_t positions[2] = {126, 127};
    for (int t = 0; t < 2; t++) {
        TEST_ASSERT(ds4_gpu_tensor_write(pos, 0, &positions[t], 4));
        TEST_ASSERT(ds4_gpu_v41_rope_tensor(x[t], pos, 1, 1, D, 64, 10000.f, 4096, 40.f, 32.f, 1.f, 0));
        TEST_ASSERT(test_v41_read(x[t], expected[t], sizeof input));
        TEST_ASSERT(ds4_gpu_tensor_write(x[t], 0, input, sizeof input));
    }
    TEST_ASSERT(ds4_gpu_begin_commands());
    for (int t = 0; t < 2; t++) {
        int32_t scratch = positions[t];
        TEST_ASSERT(ds4_gpu_tensor_write_zerocopy(pos, 0, &scratch, 4));
        scratch = -999; /* host source may change before GPU execution */
        TEST_ASSERT(ds4_gpu_v41_rope_tensor(x[t], pos, 1, 1, D, 64, 10000.f, 4096, 40.f, 32.f, 1.f, 0));
    }
    TEST_ASSERT(ds4_gpu_end_commands());
    for (int t = 0; t < 2; t++) {
        TEST_ASSERT(test_v41_read(x[t], got[t], sizeof input));
        test_v41_cmp("plain decode posg RoPE", got[t], expected[t], D, 0.f, 0.f);
        ds4_gpu_tensor_free(x[t]);
    }
    ds4_gpu_tensor_free(pos);
}

static void test_staging_reuse(void) {
    const uint32_t N = 8192; /* two pages in one uncommitted batch */
    ds4_gpu_tensor *src = test_v41_tensor(NULL, 16), *dst = test_v41_tensor(NULL, N * 4);
    uint32_t *got = malloc(N * 4);
    uint64_t warm = 0;
    for (uint32_t cycle = 0; cycle < 5; cycle++) {
        TEST_ASSERT(ds4_gpu_begin_commands());
        for (uint32_t i = 0; i < N; i++) {
            uint32_t value = i + cycle * N;
            TEST_ASSERT(ds4_gpu_tensor_write(src, 4, &value, 4));
            TEST_ASSERT(ds4_gpu_tensor_copy(dst, i * 4, src, 4, 4));
        }
        TEST_ASSERT(ds4_gpu_end_commands());
        TEST_ASSERT(test_v41_read(dst, got, N * 4));
        int bad = 0;
        for (uint32_t i = 0; i < N; i++) if (got[i] != i + cycle * N) bad++;
        TEST_ASSERT(bad == 0);
        uint64_t bytes = ds4_gpu_current_allocated_bytes();
        if (cycle == 0) warm = bytes;
        else TEST_ASSERT(bytes <= warm + 4u * 1024u * 1024u);
    }
    /* Unaligned copy through a tensor view exercises the blit fallback. */
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(src, 1, 7);
    const unsigned char want[3] = {19, 31, 73}; unsigned char readback[3];
    TEST_ASSERT(ds4_gpu_begin_commands());
    TEST_ASSERT(ds4_gpu_tensor_write(view, 1, want, 3));
    TEST_ASSERT(ds4_gpu_tensor_copy(dst, 1, view, 1, 3));
    TEST_ASSERT(ds4_gpu_end_commands());
    TEST_ASSERT(ds4_gpu_tensor_read(dst, 1, readback, 3));
    TEST_ASSERT(memcmp(want, readback, 3) == 0);
    ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(src); ds4_gpu_tensor_free(dst); free(got);
}

void test_metal_upload(void) {
    TEST_ASSERT(ds4_gpu_init());
    test_ordered_upload();
    test_decode_positions();
    test_staging_reuse();
}
#else
typedef int ds4_t_metal_upload_nonempty_tu;
#endif
