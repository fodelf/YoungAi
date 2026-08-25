/* pubbench_extract_test.c — pubbench.c 抽取器金标回归(2026-08-25 Python→C 迁移)。
 * 夹具 pubbench_fixtures/cases.jsonl(构造样例) + expect.jsonl(.py 版 extract_completion
 * 的输出, 生成于迁移当日)。本驱动只跑抽取器, 不碰 server/子进程, 秒级出判决。
 * 构建: gcc -O3 -o pubbench_extract_test pubbench_extract_test.c -lcurl -lz -lpthread -lm
 * 用法: ./pubbench_extract_test [夹具目录]   (默认 = 本文件同目录的 pubbench_fixtures) */
#define main pubbench_main_unused
#include "../pubbench.c"
#undef main

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "pubbench_fixtures";
    char pc[4096], pe[4096];
    snprintf(pc, sizeof pc, "%s/cases.jsonl", dir);
    snprintf(pe, sizeof pe, "%s/expect.jsonl", dir);
    rows_t cs = load_jsonl(pc), ex = load_jsonl(pe);
    if (cs.n != ex.n) { fprintf(stderr, "夹具行数不匹配 %zu vs %zu\n", cs.n, ex.n); return 2; }
    int bad = 0;
    for (size_t i = 0; i < cs.n; i++) {
        jv *c = cs.row[i];
        const char *name = jstr_req(c, "name");
        const char *lang = jstr_req(c, "lang");
        const char *const *stops = strcmp(lang, "python") ? GO_STOP : PY_STOP;
        jv *sp = jget(c, "strip_prompt"), *cb = jget(c, "close_brace");
        str_t got = extract_completion(jstr_req(c, "text"), jstr_req(c, "prompt"), stops, sp->b, cb->b);
        jv *want = jget(ex.row[i], "out");
        if (strcmp(jstr_req(ex.row[i], "name"), name)) { fprintf(stderr, "夹具错位 @%zu\n", i); return 2; }
        if (got.n != want->slen || memcmp(got.p, want->s, got.n)) {
            bad++;
            printf("MISMATCH %-26s\n  want(%zu): ", name, want->slen);
            for (size_t k = 0; k < want->slen; k++) printf(want->s[k] == '\n' ? "\\n" : "%c", want->s[k]);
            printf("\n  got (%zu): ", got.n);
            for (size_t k = 0; k < got.n; k++) printf(got.p[k] == '\n' ? "\\n" : "%c", got.p[k]);
            printf("\n");
        } else {
            printf("ok       %-26s (%zu B)\n", name, got.n);
        }
        free(got.p);
    }
    printf("[extract] %zu/%zu 逐字节一致\n", cs.n - bad, cs.n);
    return bad ? 1 : 0;
}
