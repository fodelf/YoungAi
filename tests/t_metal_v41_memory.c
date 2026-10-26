/* Resource lifetime regression: the C server worker has no outer ObjC pool. */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)
/* The server test header also declares its own public now_sec(). */
#define now_sec core_test_now_sec
#include "../src/core/core_internal.h"
#undef now_sec

static void test_cached_workspace(void) {
    const ds4_shape old_shape = g_ds4_shape;
    const ds4_v41_cfg old_cfg = g_ds4_v41;
    const uint32_t old_ratio = g_ds4_compress_ratios[0];
    g_ds4_shape = (ds4_shape){ .variant=DS4_VARIANT_V41, .n_layer=2, .n_embd=32, .n_vocab=16,
        .n_head=1, .n_head_dim=32, .n_out_group=1, .n_lora_q=16, .n_lora_o=16,
        .n_expert=4, .n_expert_used=1, .n_ff_exp=32, .n_swa=128,
        .n_indexer_head=1, .n_indexer_head_dim=32, .n_indexer_top_k=4, .n_hc=4 };
    g_ds4_v41 = (ds4_v41_cfg){ .active=1, .ctx=1048576, .mtp_block=5, .n_mtp_target=1,
        .is_kv_source={1}, .candidate_source_layer=-1 };
    g_ds4_compress_ratios[0] = 4;
    int32_t *prompt = calloc(g_ds4_v41.ctx, 4);
    TEST_ASSERT(prompt != NULL);
    const uint32_t lengths[] = {70754, 524288, 1048560};
    for (unsigned li = 0; li < sizeof lengths / sizeof lengths[0]; li++) {
        const uint32_t np = lengths[li], ctx = g_ds4_v41.ctx;
        ds4_v41_state src, dst;
        v41_kvreuse_drop();
        TEST_ASSERT(v41_state_alloc(&src, 10, ctx, 10));
        src.n_past = np; src.ng_src[0] = np / 4;
        src.mainh_n = 128; src.mainh_end = (int64_t)np - 1;
        TEST_ASSERT(ds4_gpu_tensor_fill_f32(src.comp_kv[0], 0, ds4_gpu_tensor_bytes(src.comp_kv[0]) / 4));
        TEST_ASSERT(ds4_gpu_tensor_fill_f32(src.index_k[0], 0, ds4_gpu_tensor_bytes(src.index_k[0]) / 4));
        TEST_ASSERT(ds4_gpu_tensor_fill_f32(src.mainh, 3, ds4_gpu_tensor_bytes(src.mainh) / 4));
        TEST_ASSERT(ds4_gpu_tensor_fill_f32(src.logits, 7, ds4_gpu_tensor_bytes(src.logits) / 4));
        const uint64_t tail = (uint64_t)src.ng_src[0] * DS4_V41_CKV_BYTES - 4;
        const float marker = 42;
        TEST_ASSERT(ds4_gpu_tensor_write(src.comp_kv[0], tail, &marker, 4));
        v41_kvreuse_save(&src, prompt, np, NULL);
        v41_state_free(&src);
        for (uint32_t extra = 0; extra <= 5; extra += 5) {
            uint32_t reused = 0;
            TEST_ASSERT(v41_kvreuse_alloc(&dst, prompt, np + extra, ctx, 2048, NULL, &reused));
            TEST_ASSERT(reused == np && dst.n_past == np && dst.ctx == ctx);
            TEST_ASSERT(dst.cap_tok == DS4_MTP_MAX_BLOCK + 2u);
            TEST_ASSERT(dst.mainh_n == 128 && dst.mainh_end == (int64_t)np - 1);
            float value = 0;
            TEST_ASSERT(ds4_gpu_tensor_read(dst.comp_kv[0], tail, &value, 4) && value == marker);
            if (!extra) TEST_ASSERT(ds4_gpu_tensor_read(dst.logits, 0, &value, 4) && value == 7);
            v41_state_free(&dst);
        }
        /* Simulate a checkpoint that cannot be copied (missing layer 2). A
         * partially restored state must be discarded before cold prefill. */
        g_ds4_shape.n_layer = 3;
        uint32_t reused = 0;
        TEST_ASSERT(v41_kvreuse_alloc(&dst, prompt, np + 5, ctx, 2048, NULL, &reused));
        TEST_ASSERT(reused == 0 && dst.n_past == 0 && dst.cap_tok == 2048);
        float value = 1;
        TEST_ASSERT(ds4_gpu_tensor_read(dst.win[0], 0, &value, 4) && value == 0);
        v41_state_free(&dst);
        g_ds4_shape.n_layer = 2;
        fprintf(stderr, "ds4: [v41-test] prefix %u: exact / +5 reuse and failed-copy cold fallback passed\n", np);
    }
    v41_kvreuse_drop();
    /* Decode-row reclamation must preserve the context and all recurrent state.
     * It may discard logits only after generation has selected the first token. */
    ds4_v41_state shrink;
    TEST_ASSERT(v41_state_alloc(&shrink, 128, 1048576, 10));
    shrink.n_past = 70754; shrink.ng_src[0] = 70754 / 4;
    shrink.mainh_n = 128; shrink.mainh_end = 70753;
    ds4_gpu_tensor *kv = shrink.comp_kv[0], *index = shrink.index_k[0], *mainh = shrink.mainh;
    TEST_ASSERT(ds4_gpu_tensor_fill_f32(shrink.win[0], 11, 128 * 32));
    TEST_ASSERT(ds4_gpu_tensor_fill_f32(shrink.cpre_kv[0], 13, 4 * 32));
    TEST_ASSERT(v41_state_decode_shrink(&shrink, 9));
    TEST_ASSERT(shrink.cap_tok == 9 && shrink.ctx == 1048576 && shrink.n_past == 70754);
    TEST_ASSERT(shrink.comp_kv[0] == kv && shrink.index_k[0] == index && shrink.mainh == mainh);
    TEST_ASSERT(shrink.mainh_n == 128 && shrink.mainh_end == 70753 && shrink.ng_src[0] == 70754 / 4);
    TEST_ASSERT(shrink.logits && shrink.logits_rows == 9 && shrink.last_logit_row == 0);
    float check[128 * 32];
    TEST_ASSERT(ds4_gpu_synchronize());
    TEST_ASSERT(ds4_gpu_tensor_read(shrink.win[0], 0, check, sizeof check));
    for (unsigned i = 0; i < 128 * 32; i++) TEST_ASSERT(check[i] == 11);
    TEST_ASSERT(ds4_gpu_tensor_read(shrink.cpre_kv[0], 0, check, 4 * 32 * 4));
    for (unsigned i = 0; i < 4 * 32; i++) TEST_ASSERT(check[i] == 13);
    TEST_ASSERT(v41_state_decode_shrink(&shrink, 1));
    TEST_ASSERT(shrink.cap_tok == 1 && shrink.logits_rows == 1);
    TEST_ASSERT(ds4_gpu_synchronize());
    v41_state_free(&shrink);
    /* A bad embedding range rejects the forward after begin_commands. It
     * must close that batch so the next request can submit GPU work. */
    ds4_v41_state st;
    TEST_ASSERT(v41_state_alloc(&st, 10, 128, 10));
    ds4_engine e = {0}; ds4_tensor bad_weight = {0};
    e.weights.token_embd = &bad_weight;
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT(!v41_forward(&e, &st, prompt, 1));
        TEST_ASSERT(ds4_gpu_begin_commands());
        TEST_ASSERT(ds4_gpu_end_commands());
    }
    v41_state_free(&st);
    free(prompt);
    g_ds4_shape = old_shape; g_ds4_v41 = old_cfg; g_ds4_compress_ratios[0] = old_ratio;
}

void test_metal_v41_memory(void) { test_cached_workspace(); }

#else
typedef int ds4_t_metal_v41_memory_nonempty_tu;
#endif
