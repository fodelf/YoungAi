/* server_tests_kv2.c — 机械拆分自 ds4_server.c 内嵌测试块 (16619-16941 行): KV 缓存逐出。 */

#include "server_tests_internal.h"

void test_kv_cache_eviction_values_fresh_snapshots(void) {
    char tmpl[] = "/tmp/ds4-kv-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_UNKNOWN, 512, 0, now, 4096);
    test_kv_stub_file(dir, new_sha, KV_REASON_UNKNOWN, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

void test_kv_cache_eviction_prefers_anchor_reason(void) {
    char tmpl[] = "/tmp/ds4-kv-anchor-reason-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *anchor_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, anchor_sha, KV_REASON_COLD, 2048, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char anchor_name[44], continued_name[44];
    snprintf(anchor_name, sizeof(anchor_name), "%.40s.kv", anchor_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *anchor_path = path_join(dir, anchor_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(anchor_path, F_OK) == 0);
    TEST_ASSERT(access(continued_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(anchor_path);
    unlink(continued_path);
    free(anchor_path);
    free(continued_path);
    rmdir(dir);
}

void test_kv_cache_eviction_makes_room_before_store(void) {
    char tmpl[] = "/tmp/ds4-kv-pre-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 2048);

    char old_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    char *old_path = path_join(dir, old_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 4096u) + 16u;
    kv_cache_evict(&kc, NULL, KV_CACHE_FIXED_HEADER + 4u + 4096u, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(old_path);
    free(old_path);
    rmdir(dir);
}

void test_kv_cache_eviction_ignores_oversize_incoming(void) {
    char tmpl[] = "/tmp/ds4-kv-oversize-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 1024);

    char old_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    char *old_path = path_join(dir, old_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 1024u) + 16u;
    kv_cache_evict(&kc, NULL, kc.budget_bytes + 1, NULL);

    TEST_ASSERT(access(old_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    free(old_path);
    rmdir(dir);
}

void test_kv_cache_eviction_prefers_superseded_continued_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-prefix-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *continued_text = "system: hello world";
    const char *cold_text = "different stable prefix";
    const char *incoming_text = "system: hello world\nuser: prompt";
    test_kv_text_stub_file(dir, continued_text, KV_REASON_CONTINUED, 4096, 2048);
    test_kv_text_stub_file(dir, cold_text, KV_REASON_COLD, 1024, 2048);

    char continued_sha[41], cold_sha[41];
    sha1_bytes_hex(continued_text, strlen(continued_text), continued_sha);
    sha1_bytes_hex(cold_text, strlen(cold_text), cold_sha);
    char continued_name[44], cold_name[44];
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *continued_path = path_join(dir, continued_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    uint64_t incoming_bytes =
        KV_CACHE_FIXED_HEADER + 4u + strlen(incoming_text) + 2048u;
    kc.budget_bytes =
        incoming_bytes + KV_CACHE_FIXED_HEADER + 4u + strlen(cold_text) + 2048u;
    ds4_kvstore_eviction_context incoming = {
        .text = incoming_text,
        .text_len = strlen(incoming_text),
        .model_id = 0,
        .quant_bits = 2,
        .ctx_size = 32768,
        .reject_different_quant = false,
    };
    kv_cache_evict(&kc, NULL, incoming_bytes, &incoming);

    TEST_ASSERT(access(continued_path, F_OK) != 0);
    TEST_ASSERT(access(cold_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(continued_path);
    unlink(cold_path);
    free(continued_path);
    free(cold_path);
    rmdir(dir);
}

void test_kv_cache_eviction_keeps_smaller_context_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-prefix-ctx-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *continued_text = "system: hello world";
    const char *cold_text = "different stable prefix";
    const char *incoming_text = "system: hello world\nuser: prompt";
    test_kv_text_stub_file(dir, continued_text, KV_REASON_CONTINUED, 4096, 2048);
    test_kv_text_stub_file(dir, cold_text, KV_REASON_COLD, 1024, 2048);

    char continued_sha[41], cold_sha[41];
    sha1_bytes_hex(continued_text, strlen(continued_text), continued_sha);
    sha1_bytes_hex(cold_text, strlen(cold_text), cold_sha);
    char continued_name[44], cold_name[44];
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *continued_path = path_join(dir, continued_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    uint64_t incoming_bytes =
        KV_CACHE_FIXED_HEADER + 4u + strlen(incoming_text) + 2048u;
    kc.budget_bytes =
        incoming_bytes + KV_CACHE_FIXED_HEADER + 4u + strlen(continued_text) + 2048u;
    ds4_kvstore_eviction_context incoming = {
        .text = incoming_text,
        .text_len = strlen(incoming_text),
        .model_id = 0,
        .quant_bits = 2,
        .ctx_size = 65536,
        .reject_different_quant = false,
    };
    kv_cache_evict(&kc, NULL, incoming_bytes, &incoming);

    TEST_ASSERT(access(continued_path, F_OK) == 0);
    TEST_ASSERT(access(cold_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(continued_path);
    unlink(cold_path);
    free(continued_path);
    free(cold_path);
    rmdir(dir);
}

void test_kv_cache_eviction_score_decays_stale_hits(void) {
    /* stale: lower tokens-per-byte (e.g. tool-heavy prompt) but boosted by
     * 10 hits well in the past.  fresh: higher tokens-per-byte and zero hits,
     * just stored.  The stale hit bonus decays by inactivity, so fresh wins on
     * its better baseline even though stale once had more successful hits. */
    const uint64_t now = 1000u + 14u * KV_CACHE_HIT_HALF_LIFE_SECONDS;
    kv_entry stale = {.tokens = 1024, .hits = 10, .file_size = 4096, .last_used = 1000};
    kv_entry fresh = {.tokens = 2048, .hits = 0,  .file_size = 4096, .last_used = now};

    double s_on = kv_entry_eviction_score(&stale, NULL, now, NULL);
    double f_on = kv_entry_eviction_score(&fresh, NULL, now, NULL);
    TEST_ASSERT(s_on < f_on);

    /* A fresh entry's score never decays below its (0+1) * tokens/size floor,
     * regardless of how old another entry's hit history is. */
    TEST_ASSERT(f_on == 1.0 * (double)fresh.tokens / (double)fresh.file_size);
}

void test_kv_cache_eviction_decayed_hits_tie_break_by_age(void) {
    char tmpl[] = "/tmp/ds4-kv-stale-hit-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    uint64_t stale = now > KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        ? now - KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        : 1;
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 2048, 15, stale, 2048);
    test_kv_stub_file(dir, new_sha, KV_REASON_COLD, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

void test_kv_cache_eviction_keeps_aligned_continued_frontiers(void) {
    char tmpl[] = "/tmp/ds4-kv-live-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *cold_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 512, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char cold_name[44], continued_name[44];
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *cold_path = path_join(dir, cold_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(cold_path, F_OK) != 0);
    TEST_ASSERT(access(continued_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(cold_path);
    unlink(continued_path);
    free(cold_path);
    free(continued_path);
    rmdir(dir);
}

static void kv_conv_stub_file(const char *dir, const char *text, const char *conv,
                              uint64_t payload_bytes) {
    char sha[41], name[44];
    sha1_bytes_hex(text, strlen(text), sha);
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) { free(path); return; }
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, 0, 2, DS4_KVSTORE_REASON_COLD, 0, 512, 0,
                            32768, 100, 100, payload_bytes);
    uint8_t text_len[4];
    le_put32(text_len, (uint32_t)strlen(text));
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
    for (uint64_t i = 0; i < payload_bytes; i++) TEST_ASSERT(fputc(0, fp) != EOF);
    TEST_ASSERT(ds4_kvstore_conv_write(fp, conv) == DS4_KVSTORE_CONV_TRAILER);
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}

void test_kv_cache_conv_trailer_roundtrip(void) {
    char tmpl[] = "/tmp/ds4-kv-conv-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *text = "conv-owner transcript", *conv = "agent-alpha";
    uint8_t cv_short[DS4_KVSTORE_CONV_TRAILER], cv_long[DS4_KVSTORE_CONV_TRAILER];
    ds4_kvstore_conv_trailer(cv_short, "abc");
    ds4_kvstore_conv_trailer(cv_long, "abcdef");
    TEST_ASSERT(ds4_kvstore_conv_match(cv_short, "abc"));
    TEST_ASSERT(!ds4_kvstore_conv_match(cv_short, "abcdef"));   /* 60B zero pad makes short names not prefixes */
    TEST_ASSERT(!ds4_kvstore_conv_match(cv_long, "abc"));
    kv_conv_stub_file(dir, text, conv, 16);

    kv_disk_cache kc = {0};
    TEST_ASSERT(ds4_kvstore_open(&kc, dir, 1, false,
                                 ds4_kvstore_default_options(), "test", NULL, NULL));
    TEST_ASSERT(kc.len == 1);
    TEST_ASSERT(kc.len == 1 && strcmp(kc.entry[0].conv, conv) == 0);
    ds4_kvstore_close(&kc);

    char sha[41], name[44];
    sha1_bytes_hex(text, strlen(text), sha);
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    unlink(path);
    free(path);
    rmdir(dir);
}

void test_kv_cache_erase_conversation_keeps_others(void) {
    char tmpl[] = "/tmp/ds4-kv-conv-erase-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *a = "alpha transcript", *b = "beta transcript";
    kv_conv_stub_file(dir, a, "agent-alpha", 16);
    kv_conv_stub_file(dir, b, "agent-beta", 16);

    kv_disk_cache kc = {0};
    TEST_ASSERT(ds4_kvstore_open(&kc, dir, 1, false,
                                 ds4_kvstore_default_options(), "test", NULL, NULL));
    TEST_ASSERT(kc.len == 2);
    TEST_ASSERT(ds4_kvstore_erase_conversation(&kc, "agent-alpha") == 1);
    TEST_ASSERT(kc.len == 1);
    TEST_ASSERT(kc.len == 1 && strcmp(kc.entry[0].conv, "agent-beta") == 0);

    char sha_a[41], sha_b[41], name[44], *path_a, *path_b;
    sha1_bytes_hex(a, strlen(a), sha_a);
    sha1_bytes_hex(b, strlen(b), sha_b);
    snprintf(name, sizeof(name), "%.40s.kv", sha_a);
    path_a = path_join(dir, name);
    TEST_ASSERT(access(path_a, F_OK) != 0);   /* alpha removed from disk */
    snprintf(name, sizeof(name), "%.40s.kv", sha_b);
    path_b = path_join(dir, name);
    TEST_ASSERT(access(path_b, F_OK) == 0);   /* beta kept for the other agent */
    ds4_kvstore_close(&kc);
    unlink(path_b);
    free(path_a);
    free(path_b);
    rmdir(dir);
}
