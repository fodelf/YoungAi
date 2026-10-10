/* train_gen.c — 出题: 文档块 → 问答料(2026-10-10, 原 z_nightly_spark.sh stage_gen + kd_split(python) 的 C 版)。总述见 train_internal.h。
 * 这是训练链外的可选数据工具(10-10 用户令对齐 unsloth: 训练器只收 messages / text 两种料, 出题 = 它的 synthetic data 工具, 零领域措辞):
 *   一份作业 = chunks/<块>.txt 原文 + 一句通用种子(照 Meta synthetic-data-kit 默认 qa_generation 译成中文; <料目录>/seeds/qa.txt 可覆盖),
 *   合批交给 ./ds4 --gen-jobs(cli_gen_jobs.c, 引擎写 gen/<块>_s0rN.txt; 没收口的不落盘), 产物已在的跳过(断点续跑)。
 *   拆对: 认 "问：/答："(前后可带序号) 的行, 同块同题去重, 每题带 context = 块原文 → <料目录>/<料名>.jsonl, 训练器按 context 做上下文蒸馏。
 * 唯一加的要求"问题自带完整指代"是蒸馏结构的约束(学生训练时不看材料, "这段文字"指不到东西), 不是领域味。 */
#include "train_internal.h"
#include <dirent.h>
#include <sys/stat.h>

#define TR_GEN_MAX_TOK 4096   /* 一份作业的生成上限: 10 组问答远用不完, 只防跑飞 */
static const char *const TR_SEED_QA =
    "请根据上面这段文字写 10 组问答，用于训练大模型。规则：1. 问题只问这段文字里的重要事实；2. 答案必须能在这段文字里直接找到依据；"
    "3. 每个问题写明完整指代（具体的名称、日期、对象），不许出现“这段文字”“上文”之类的说法。严格按下面的格式输出，不要输出别的内容：\n问：……\n答：……";

static bool exists(const char *p) { struct stat st; return stat(p, &st) == 0 && st.st_size > 0; }
static int by_name(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
/* chunks/ 下的 .txt 块名(去扩展名), 排好序 */
static int list_chunks(const char *doc_dir, char ***out) {
    char d[TR_PATH + 320]; snprintf(d, sizeof d, "%s/chunks", doc_dir);
    DIR *dd = opendir(d); struct dirent *de; char **v = NULL; int n = 0;
    while (dd && (de = readdir(dd))) {
        const size_t L = strlen(de->d_name);
        if (L <= 4 || strcmp(de->d_name + L - 4, ".txt")) continue;
        v = realloc(v, (size_t)(n + 1) * sizeof *v); v[n] = strdup(de->d_name); v[n][L - 4] = 0; n++;
    }
    if (dd) closedir(dd);
    if (n) qsort(v, (size_t)n, sizeof *v, by_name);
    *out = v;
    return n;
}
static unsigned cksum_seed(const char *s) { unsigned h = 2166136261u; for (; *s; s++) h = (h ^ (unsigned char)*s) * 16777619u; return h ? h : 1; }

bool tr_gen_jobs(const char *doc_dir, int rounds, char *jobs_path, size_t jn, unsigned *njobs, FILE *log) {
    char **ch; const int nc = list_chunks(doc_dir, &ch);
    *njobs = 0;
    if (!nc) { tr_logf(log, "★%s/chunks 里没有 .txt 块★", doc_dir); return false; }
    char p[TR_PATH + 400], seedp[TR_PATH + 340];
    snprintf(p, sizeof p, "%s/gen", doc_dir); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/gen/prompt", doc_dir); mkdir(p, 0755);
    snprintf(seedp, sizeof seedp, "%s/seeds/qa.txt", doc_dir);
    char *seed = tr_slurp(seedp, NULL);
    snprintf(jobs_path, jn, "%s/gen/jobs.tsv", doc_dir);
    FILE *jf = fopen(jobs_path, "w");
    if (!jf) { free(seed); return false; }
    for (int r = 1; r <= rounds; r++) for (int i = 0; i < nc; i++) {
        char name[320], outp[TR_PATH + 400], outj[TR_PATH + 400], pr[TR_PATH + 400], cp[TR_PATH + 400];
        snprintf(name, sizeof name, "%s_s0r%d", ch[i], r);   /* 名字留 _s0: 老目录的 s1/s2 产物同一规则拆 */
        snprintf(outp, sizeof outp, "%s/gen/%s.txt", doc_dir, name); snprintf(outj, sizeof outj, "%s/gen/%s.json", doc_dir, name);
        if (exists(outp) || exists(outj)) continue;
        snprintf(cp, sizeof cp, "%s/chunks/%s.txt", doc_dir, ch[i]);
        char *text = tr_slurp(cp, NULL); if (!text) continue;
        snprintf(pr, sizeof pr, "%s/gen/prompt/%s.txt", doc_dir, name);
        FILE *pf = fopen(pr, "w");
        if (pf) { fputs(text, pf); fprintf(pf, "\n%s", seed ? seed : TR_SEED_QA); fclose(pf); fprintf(jf, "%s\t%s\t%u\t%d\n", pr, outp, cksum_seed(name), TR_GEN_MAX_TOK); (*njobs)++; }
        free(text);
    }
    fclose(jf); free(seed);
    tr_logf(log, "出题: %d 块 × %d 轮, 本次要生成 %u 份 → %s/gen", nc, rounds, *njobs, doc_dir);
    for (int i = 0; i < nc; i++) free(ch[i]);
    free(ch);
    return true;
}

/* "问：xx" / "1. 答2：xx" 这类标记行: 返回 'q'/'a'/0, *rest 指向标记后的正文 */
static char mark_of(const char *ln, const char **rest) {
    const char *p = ln;
    while (*p == ' ' || *p == '\t') p++;
    if (*p >= '0' && *p <= '9') { while (*p >= '0' && *p <= '9') p++; if (*p == '.' || *p == ')' || !strncmp(p, "、", 3)) p += (*p == '.' || *p == ')') ? 1 : 3; while (*p == ' ') p++; }
    char kind = 0;
    if (!strncmp(p, "问", 3)) kind = 'q'; else if (!strncmp(p, "答", 3)) kind = 'a'; else return 0;
    p += 3;
    while (*p == ' ') p++;
    while (*p >= '0' && *p <= '9') p++;
    while (*p == ' ') p++;
    if (!strncmp(p, "：", 3)) p += 3; else if (*p == ':') p++; else return 0;
    while (*p == ' ') p++;
    *rest = p;
    return kind;
}
static void trim(ds4_buf *b) { while (b->len && (b->ptr[b->len - 1] == '\n' || b->ptr[b->len - 1] == ' ')) b->ptr[--b->len] = 0; }
/* 去掉 ** 加粗与空白后的题干, 同块去重用 */
static char *qkey(const char *q) { char *k = malloc(strlen(q) + 1), *o = k; for (; *q; q++) if (*q != ' ' && *q != '\n' && *q != '\t' && *q != '*') *o++ = *q; *o = 0; return k; }

typedef struct { char **keys; int n; } seen_t;
static bool seen_add(seen_t *s, const char *q) {
    char *k = qkey(q);
    for (int i = 0; i < s->n; i++) if (!strcmp(s->keys[i], k)) { free(k); return false; }
    s->keys = realloc(s->keys, (size_t)(s->n + 1) * sizeof *s->keys); s->keys[s->n++] = k;
    return true;
}
static void strip_bold(ds4_buf *b) {   /* 模型爱把题干加粗, 料里不要 markdown */
    if (!b->ptr) return;
    char *o = b->ptr; for (const char *p = b->ptr; *p; ) { if (p[0] == '*' && p[1] == '*') { p += 2; continue; } *o++ = *p++; }
    *o = 0; b->len = (size_t)(o - b->ptr);
}
static void emit_pair(FILE *out, ds4_buf *q, ds4_buf *a, const char *ctx, seen_t *seen, unsigned *n) {
    strip_bold(q); strip_bold(a); trim(q); trim(a);
    if (!q->len || !a->len || !seen_add(seen, q->ptr)) return;
    ds4_buf b = {0};
    ds4_buf_puts(&b, "{\"messages\": [{\"role\": \"user\", \"content\": "); ds4_json_escape(&b, q->ptr);
    ds4_buf_puts(&b, "}, {\"role\": \"assistant\", \"content\": "); ds4_json_escape(&b, a->ptr);
    ds4_buf_puts(&b, "}], \"context\": "); ds4_json_escape(&b, ctx); ds4_buf_puts(&b, "}\n");
    fwrite(b.ptr, 1, b.len, out); ds4_buf_free(&b); (*n)++;
}

unsigned tr_gen_split(const char *doc_dir, FILE *log) {
    char **ch; const int nc = list_chunks(doc_dir, &ch);
    const char *base = strrchr(doc_dir, '/'); base = base ? base + 1 : doc_dir;
    char outp[TR_PATH + 400]; snprintf(outp, sizeof outp, "%s/%s.jsonl", doc_dir, base);
    FILE *out = fopen(outp, "w"); unsigned n = 0, bad = 0;
    if (!out) return 0;
    for (int i = 0; i < nc; i++) {
        char cp[TR_PATH + 400]; snprintf(cp, sizeof cp, "%s/chunks/%s.txt", doc_dir, ch[i]);
        char *ctx = tr_slurp(cp, NULL); if (!ctx) continue;
        { size_t L = strlen(ctx); while (L && (ctx[L - 1] == '\n' || ctx[L - 1] == ' ')) ctx[--L] = 0; }
        seen_t seen = { NULL, 0 };
        char gd[TR_PATH + 330]; snprintf(gd, sizeof gd, "%s/gen", doc_dir);
        DIR *d = opendir(gd); struct dirent *de; char **files = NULL; int nf = 0;
        const size_t cl = strlen(ch[i]);
        while (d && (de = readdir(d))) {   /* 本块的产物: <块>_s<n>r<n>.txt(老目录的 .json 不再认: 引擎写的原始 .txt 才是现役格式) */
            const size_t L = strlen(de->d_name);
            if (L <= cl + 4 || strncmp(de->d_name, ch[i], cl) || de->d_name[cl] != '_' || de->d_name[cl + 1] != 's' || strcmp(de->d_name + L - 4, ".txt")) continue;
            files = realloc(files, (size_t)(nf + 1) * sizeof *files); files[nf++] = strdup(de->d_name);
        }
        if (d) closedir(d);
        if (nf) qsort(files, (size_t)nf, sizeof *files, by_name);
        for (int f = 0; f < nf; f++) {
            char fp[TR_PATH + 400]; snprintf(fp, sizeof fp, "%s/gen/%s", doc_dir, files[f]);
            char *t = tr_slurp(fp, NULL); free(files[f]);
            if (!t) continue;
            ds4_buf q = {0}, a = {0}; char mode = 0; unsigned before = n;
            char *sv = NULL;
            for (char *ln = strtok_r(t, "\n", &sv); ln; ln = strtok_r(NULL, "\n", &sv)) {
                const char *rest; const char m = mark_of(ln, &rest);
                if (m == 'q') { emit_pair(out, &q, &a, ctx, &seen, &n); q.len = 0; a.len = 0; if (q.ptr) q.ptr[0] = 0; if (a.ptr) a.ptr[0] = 0; ds4_buf_puts(&q, rest); mode = 'q'; }
                else if (m == 'a') { a.len = 0; if (a.ptr) a.ptr[0] = 0; ds4_buf_puts(&a, rest); mode = 'a'; }
                else if (mode == 'q') { ds4_buf_putc(&q, '\n'); ds4_buf_puts(&q, ln); }
                else if (mode == 'a') { ds4_buf_putc(&a, '\n'); ds4_buf_puts(&a, ln); }
            }
            emit_pair(out, &q, &a, ctx, &seen, &n);
            if (n == before) bad++;
            ds4_buf_free(&q); ds4_buf_free(&a); free(t);
        }
        free(files);
        for (int k = 0; k < seen.n; k++) free(seen.keys[k]);
        free(seen.keys); free(ctx);
    }
    fclose(out);
    for (int i = 0; i < nc; i++) free(ch[i]);
    free(ch);
    tr_logf(log, "问答拆对: %d 块, %u 题 → %s; 没拆出问答的生成 %u 份", nc, n, outp, bad);
    return n;
}
