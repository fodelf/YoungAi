/* compare_scores.c — 两份官方续写评分 TSV 对比(C, 2026-08-25 Python→C 迁移)。
 * 取代 compare_scores.py, stdout 逐字节同(排序=py 元组序: delta 升序→case_id 字典序;
 * 重复 id=后者覆盖, 与 dict 语义同)。
 * 用法: compare_scores OLD.tsv NEW.tsv */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char id[128]; long tt; double nll, avg; long fm, lcp; int used; } row_t;

static int load(const char *path, row_t **out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    char ln[4096];
    if (!fgets(ln, sizeof ln, f)) exit(2);
    /* 表头列序动态解析(与 DictReader 同) */
    int ci = -1, ct = -1, cn = -1, ca = -1, cf = -1, cl = -1, nc = 0;
    { char *tok = strtok(ln, "\t\r\n");
      while (tok) { if (!strcmp(tok, "id")) ci = nc; else if (!strcmp(tok, "target_tokens")) ct = nc;
          else if (!strcmp(tok, "nll")) cn = nc; else if (!strcmp(tok, "avg_nll")) ca = nc;
          else if (!strcmp(tok, "first_match")) cf = nc; else if (!strcmp(tok, "greedy_lcp")) cl = nc;
          nc++; tok = strtok(NULL, "\t\r\n"); } }
    row_t *rows = NULL; int n = 0, cap = 0;
    while (fgets(ln, sizeof ln, f)) {
        char *cols[64]; int k = 0;
        for (char *tok = strtok(ln, "\t\r\n"); tok && k < 64; tok = strtok(NULL, "\t\r\n")) cols[k++] = tok;
        if (k <= ci || k <= ct || k <= cn || k <= ca || k <= cf || k <= cl) continue;
        row_t r; memset(&r, 0, sizeof r);
        snprintf(r.id, sizeof r.id, "%s", cols[ci]);
        r.tt = atol(cols[ct]); r.nll = atof(cols[cn]); r.avg = atof(cols[ca]);
        r.fm = atol(cols[cf]); r.lcp = atol(cols[cl]);
        int dup = -1;
        for (int i = 0; i < n; i++) if (!strcmp(rows[i].id, r.id)) { dup = i; break; }
        if (dup >= 0) rows[dup] = r;   /* dict 语义: 后者覆盖 */
        else { if (n == cap) { cap = cap ? cap * 2 : 64; rows = realloc(rows, cap * sizeof(row_t)); }
               rows[n++] = r; }
    }
    fclose(f); *out = rows; return n;
}

typedef struct { double delta; const char *id; long t; double oavg, navg; } dl_t;
static int cmp_id(const void *a, const void *b) { return strcmp(((const row_t *)a)->id, ((const row_t *)b)->id); }
static int cmp_dl(const void *a, const void *b) {
    const dl_t *x = a, *y = b;
    if (x->delta < y->delta) return -1;
    if (x->delta > y->delta) return 1;
    return strcmp(x->id, y->id);   /* py 元组序第二键 */
}
static int cmp_dl_rev(const void *a, const void *b) { return cmp_dl(b, a); }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s OLD.tsv NEW.tsv\n", argv[0]); return 2; }
    row_t *ro, *rn;
    int no = load(argv[1], &ro), nn = load(argv[2], &rn);
    qsort(ro, no, sizeof(row_t), cmp_id);
    dl_t *dl = malloc((size_t)no * sizeof(dl_t));
    long ids = 0, tokens = 0, of = 0, nf = 0, ol = 0, nl2 = 0;
    double onll = 0, nnll = 0;
    long win_n = 0, win_o = 0, ties = 0;
    for (int i = 0; i < no; i++) {   /* sorted(old∩new): ro 已按 id 排 */
        row_t *o = &ro[i], *n = NULL;
        for (int j = 0; j < nn; j++) if (!strcmp(rn[j].id, o->id)) { n = &rn[j]; break; }
        if (!n) continue;
        if (o->tt != n->tt) { fprintf(stderr, "token-count mismatch for %s\n", o->id); return 1; }
        tokens += o->tt; onll += o->nll; nnll += n->nll;
        of += o->fm; nf += n->fm; ol += o->lcp; nl2 += n->lcp;
        double delta = n->nll - o->nll;
        dl[ids] = (dl_t){delta, o->id, o->tt, o->avg, n->avg};
        if (delta < -1e-9) win_n++; else if (delta > 1e-9) win_o++; else ties++;
        ids++;
    }
    if (!ids) { fprintf(stderr, "no common cases\n"); return 1; }
    double ao = onll / tokens, an = nnll / tokens;
    printf("cases\t%ld\n", ids);
    printf("tokens\t%ld\n", tokens);
    printf("old_avg_nll\t%.9f\n", ao);
    printf("new_avg_nll\t%.9f\n", an);
    printf("delta_new_minus_old\t%.9f\n", an - ao);
    printf("relative_nll_change\t%.3f%%\n", (an / ao - 1.0) * 100.0);
    printf("case_wins_new_old_ties\t%ld\t%ld\t%ld\n", win_n, win_o, ties);
    printf("first_token_matches_old_new\t%ld\t%ld\n", of, nf);
    printf("avg_greedy_lcp_old_new\t%.3f\t%.3f\n", (double)ol / ids, (double)nl2 / ids);
    printf("\nnew best cases:\n");
    qsort(dl, ids, sizeof(dl_t), cmp_dl);
    for (long i = 0; i < 8 && i < ids; i++)
        printf("%s\tdelta_nll=%.6f\ttokens=%ld\told=%.6f\tnew=%.6f\n", dl[i].id, dl[i].delta, dl[i].t, dl[i].oavg, dl[i].navg);
    printf("\nold best cases:\n");
    qsort(dl, ids, sizeof(dl_t), cmp_dl_rev);
    for (long i = 0; i < 8 && i < ids; i++)
        printf("%s\tdelta_nll=%.6f\ttokens=%ld\told=%.6f\tnew=%.6f\n", dl[i].id, dl[i].delta, dl[i].t, dl[i].oavg, dl[i].navg);
    return 0;
}
