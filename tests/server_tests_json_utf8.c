/* server_tests_json_utf8.c — JSON 字符串转义对非法 UTF-8 的处理(2026-11-01 实撞)。
 * 现场照 trace 原样: 字节级 BPE 的合并 token 15725 = "ా" + 下一个字的前两个字节 E0 B0, 温 1 采样下模型接着采了完整字 "క"(E0 B0 95)
 * 而不是续接字节, 正文里就悬着 E0 B0。以前逐字节原样拷进 JSON, 客户端严格解码炸掉整项 IFEval 跑分。要的行为: 和 HF tokenizer
 * decode(errors="replace") 一样, 每段非法 / 不完整序列换成一个 U+FFFD, 合法的字节一个不动; 三个入口(带引号 / 定长 / 片段)同一份实现。 */
#include "server_tests_internal.h"

static void check_escape(const char *in, size_t n, const char *want_body, const char *why) {
    buf b = {0}; json_escape_n(&b, in, n);
    char want[512]; snprintf(want, sizeof want, "\"%s\"", want_body);
    TEST_ASSERT(b.ptr && strcmp(b.ptr, want) == 0);
    if (!b.ptr || strcmp(b.ptr, want)) fprintf(stderr, "  json_escape(%s): got %s want %s\n", why, b.ptr ? b.ptr : "(null)", want);
    ds4_buf_free(&b);
    buf f = {0}; json_escape_fragment_n(&f, in, n);   /* 片段入口(SSE 增量用)同一结果, 只是不带引号 */
    TEST_ASSERT(f.ptr && strcmp(f.ptr, want_body) == 0);
    ds4_buf_free(&f);
}

void test_json_escape_replaces_invalid_utf8(void) {
    const char R[] = "\xEF\xBF\xBD";
    /* ① trace 现场: "వా" + 悬着的 E0 B0 + "క" → 悬着的两个字节换一个 U+FFFD, 前后的字不动 */
    const char scene[] = "\xE0\xB0\xB5\xE0\xB0\xBE\xE0\xB0" "\xE0\xB0\x95\xE0\xB1\x8D";
    char want[64]; snprintf(want, sizeof want, "\xE0\xB0\xB5\xE0\xB0\xBE%s\xE0\xB0\x95\xE0\xB1\x8D", R);
    check_escape(scene, strlen(scene), want, "trace scene");
    /* ② 合法的 2/3/4 字节与 ASCII 转义照旧 */
    check_escape("a\"b\\c\n\t\x01" "é中😀", strlen("a\"b\\c\n\t\x01" "é中😀"), "a\\\"b\\\\c\\n\\t\\u0001é中😀", "valid mix");
    /* ③ 孤立续接字节、C0/C1 超长编码的领头、F5 以上: 每个坏字节各换一个 */
    snprintf(want, sizeof want, "x%sy%s%sz%s", R, R, R, R);
    check_escape("x\x80y\xC0\xAFz\xF5", 7, want, "lone bytes");
    /* ④ 末尾截断的三字节(E4 BD)整段换一个; 中间截断(E4 BD 后接 ASCII)同样 */
    snprintf(want, sizeof want, "ok%s", R); check_escape("ok\xE4\xBD", 4, want, "tail truncated");
    snprintf(want, sizeof want, "%sA", R);  check_escape("\xE4\xBD" "A", 3, want, "mid truncated");
    /* ⑤ E0 后第二字节 < A0(超长)和 ED 后 >= A0(代理区)不算合法: 领头换一个, 后面的续接字节各自再换 */
    snprintf(want, sizeof want, "%s%s%s", R, R, R); check_escape("\xE0\x80\x80", 3, want, "overlong E0");
    check_escape("\xED\xA0\x80", 3, want, "surrogate ED");
    /* ⑥ 定长入口 n 短于串: 只看前 n 字节, 切在半个字上也只换一个 */
    snprintf(want, sizeof want, "%s", R); check_escape("\xE4\xBD\xA0", 2, want, "n cuts a char");
}
