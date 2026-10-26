/* Model-free regressions. Keep this file usable on the parent revision: the
 * same tests must expose the old upload race and autorelease lifetime leak. */
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

static void test_reclamation(void) {
    const uint64_t bytes = 4u * 1024u * 1024u;
    float *host = calloc(1, (size_t)bytes);
    TEST_ASSERT(host != NULL);
    uint64_t baseline = 0, peak = 0;
    for (int i = 0; i < 40; i++) {
        ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes), *b = ds4_gpu_tensor_alloc(bytes);
        TEST_ASSERT(a && b);
        host[0] = (float)i;
        TEST_ASSERT(ds4_gpu_tensor_write(a, 0, host, bytes));
        TEST_ASSERT(ds4_gpu_begin_commands());
        TEST_ASSERT(ds4_gpu_tensor_copy(b, 0, a, 0, bytes));
        TEST_ASSERT(ds4_gpu_flush_commands());
        TEST_ASSERT(ds4_gpu_tensor_write(a, 0, host, bytes));
        TEST_ASSERT(ds4_gpu_tensor_copy(b, 0, a, 0, bytes));
        TEST_ASSERT(ds4_gpu_end_commands());
        TEST_ASSERT(ds4_gpu_synchronize());
        float got = -1;
        TEST_ASSERT(ds4_gpu_tensor_read(b, 0, &got, sizeof got));
        TEST_ASSERT(got == (float)i);
        ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
        const uint64_t now = ds4_gpu_current_allocated_bytes();
        if (i == 7) baseline = peak = now;
        if (i >= 8 && now > peak) peak = now;
    }
    fprintf(stderr, "ds4: [v41-test] resources after 32 warmed cycles: baseline %.2f MiB, peak %.2f MiB\n",
            (double)baseline / 1048576., (double)peak / 1048576.);
    TEST_ASSERT(peak <= baseline + 16u * 1024u * 1024u);
    free(host);
}
void test_metal_tensor_lifetime(void) {
    TEST_ASSERT(ds4_gpu_init());
    test_ordered_upload();
    test_reclamation();
}
#else
typedef int ds4_t_metal_tensor_lifetime_nonempty_tu;
#endif
