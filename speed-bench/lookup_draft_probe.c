/* lookup_draft_probe.c — "提示查表草稿"(Strata src/spec/suffix_drafter.cpp 那条)值不值得进引擎的离线探针(2026-10-07)。
 *
 * 要回答的问题: 解码出来的 token 里有多少是在"照抄提示 / 照抄前文"? 这些位置上零字节的查表草稿能换回几个 token, 折成速度是多少?
 * 怎么用:  cc -std=c99 -O2 -o lookup_draft_probe speed-bench/lookup_draft_probe.c && ./lookup_draft_probe <料>...
 *   料 = dkgen 的 ids(一行一个 token id, 提示 + 底座采样的续写, 旁边 <名>.ids.np 一个数 = 提示长), 或 <prompt.ids>:<gen.ids> 一对(kdrft 的样本)。
 * 做法(照 Strata): 当前尾 3 个 id 在可见历史里找同样的 trigram, 取向后匹配最长的那处(平局取最新), 照抄它后面的 k 个 id 当草稿。
 *   A = 只在提示里找(Strata 的默认索引范围), B = 提示 + 已生成的都找(报告里也抄过的词会重复出现)。
 * 判接受: 草稿与真实续写逐位比, 第一个不同处停。贪心文本下这就是验证结果; 温 1 采样文本下"真 token == 草稿"恰是点质量草稿接受概率的一次抽样, 均值无偏。
 * 速度模拟: 查表轮 = 验证 1+k 行、不跑草稿塔 = C1 + k·CR ms, 出 a+1 个 token(a 接受 + 1 个验证送的); 没有合格草稿的位置按现役每 token 均价 BASE 走
 *   (MTP 投机轮均摊到每个 token)。三个常量都是 fable5 10-07 的实测: C1 = 纯解码一步 33.7 ms, CR = 验证批每多一行 8.4 ms(n=4 58.9 / n=6 ~76 对上),
 *   BASE = 0908 贪心 46.5 t/s = 21.5 ms/token。门 L = 最短向后匹配长度(Strata min_match 默认 3), K = 最多草稿位(Strata 5; ds4 验证批 8 行 ⇒ 7)。
 * 读数怎么看: 覆盖 = 生成位置里有合格草稿的比例; p1 = 草稿第一位命中率; ā = 平均接受位; 份额 = 模拟里由查表轮产出的 token 占比; t/s 对 BASE 的比值才是结论。
 * 不这么量直接进引擎会怎样: 查表草稿错第一位就白付一整轮 1+k 行(75.7 ms 出 1 个 token, 是均价的 3.5 倍), 覆盖率不够高、精度不够高时整体反而变慢。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { KMAX = 7, MCAP = 64 };
static const double C1 = 33.7, CR = 8.4, BASE = 21.5;
static const int LS[] = { 3, 4, 6, 8, 12, 16 }, KS[] = { 5, 7 };
enum { NL = sizeof LS / sizeof *LS, NK = sizeof KS / sizeof *KS };

typedef struct { int *t; int n, np; char name[256]; } text;
typedef struct { int m, e, k, acc; } hit;   /* 最长向后匹配 m(0 = 没有), 匹配尾位置 e, 可抄的草稿位 k(≤ KMAX), 前 k 位里接受数 */
typedef struct { long pos, prop, first_ok, acc_sum, tok_lookup, tok_total; double ms; } stat;

static int *read_ids(const char *p, int *n) {
    FILE *f = fopen(p, "r"); if (!f) { perror(p); exit(1); }
    int cap = 4096, k = 0, v, *a = malloc(cap * sizeof *a);
    while (fscanf(f, "%d", &v) == 1) { if (k == cap) a = realloc(a, (size_t)(cap *= 2) * sizeof *a); a[k++] = v; }
    fclose(f); *n = k; return a;
}
static text load(const char *arg) {
    text x; memset(&x, 0, sizeof x);
    const char *c = strchr(arg, ':');
    if (c) {
        char pp[1024]; snprintf(pp, sizeof pp, "%.*s", (int)(c - arg), arg);
        int np, ng, *P = read_ids(pp, &np), *G = read_ids(c + 1, &ng);
        x.t = malloc((size_t)(np + ng) * sizeof *x.t); memcpy(x.t, P, (size_t)np * sizeof *P); memcpy(x.t + np, G, (size_t)ng * sizeof *G);
        x.n = np + ng; x.np = np; free(P); free(G); snprintf(x.name, sizeof x.name, "%s", c + 1);
    } else {
        char p[1100]; snprintf(p, sizeof p, "%s.np", arg);
        FILE *f = fopen(p, "r"); if (!f || fscanf(f, "%d", &x.np) != 1) { perror(p); exit(1); } fclose(f);
        x.t = read_ids(arg, &x.n); snprintf(x.name, sizeof x.name, "%s", arg);
    }
    const char *b = strrchr(x.name, '/'); if (b) memmove(x.name, b + 1, strlen(b));
    return x;
}

/* 位置 q(已出 t[0..q), 下一个真 token 是 t[q]); 匹配尾 e 只在 [2, elim] 里找(可见范围由 A/B 决定) */
static hit best_match(const text *x, int q, int elim) {
    hit h = { 0, -1, 0, 0 }; const int *t = x->t;
    for (int e = elim; e >= 2; e--) {
        if (t[e] != t[q-1] || t[e-1] != t[q-2] || t[e-2] != t[q-3]) continue;
        int m = 3; while (m < MCAP && e - m >= 0 && t[e-m] == t[q-1-m]) m++;
        if (m > h.m) { h.m = m; h.e = e; }
    }
    if (h.m) {
        int k = q - 1 - h.e; if (k > KMAX) k = KMAX; if (k > x->n - q) k = x->n - q;   /* 草稿只能抄已可见的 id */
        int a = 0; while (a < k && t[h.e+1+a] == t[q+a]) a++;
        h.k = k; h.acc = a;
    }
    return h;
}

static void walk(const text *x, const hit *H, int L, int K, stat *s) {
    for (int q = x->np; q < x->n;) {
        const hit *h = &H[q - x->np]; int k = h->k < K ? h->k : K;
        if (h->m >= L && k > 0) {
            int a = h->acc < k ? h->acc : k, adv = a + 1; if (q + adv > x->n) adv = x->n - q;
            s->tok_lookup += adv; s->tok_total += adv; s->ms += C1 + k * CR; q += adv;
        } else { s->tok_total++; s->ms += BASE; q++; }
    }
    for (int q = x->np; q < x->n; q++) {   /* 静态: 每个生成位置看一眼 */
        const hit *h = &H[q - x->np]; int k = h->k < K ? h->k : K; s->pos++;
        if (h->m >= L && k > 0) { int a = h->acc < k ? h->acc : k; s->prop++; s->first_ok += a > 0; s->acc_sum += a; }
    }
}

static void row(const char *tag, const stat *s) {
    printf("%-10s 覆盖 %5.1f%%  p1 %5.1f%%  ā %4.2f  份额 %5.1f%%  模拟 %5.1f t/s (×%.3f)\n", tag,
           s->pos ? 100.0 * s->prop / s->pos : 0, s->prop ? 100.0 * s->first_ok / s->prop : 0, s->prop ? (double)s->acc_sum / s->prop : 0,
           s->tok_total ? 100.0 * s->tok_lookup / s->tok_total : 0, s->ms > 0 ? 1000.0 * s->tok_total / s->ms : 0,
           s->ms > 0 ? (1000.0 * s->tok_total / s->ms) / (1000.0 / BASE) : 0);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <x.ids(带 .np)|prompt.ids:gen.ids>...\n", argv[0]); return 2; }
    static stat agg[2][NL][NK]; static long mb_prop[2][4], mb_ok[2][4];   /* Strata 的匹配长度桶: 3~5 / 6~11 / 12~23 / 24+ */
    long gen_total = 0;
    printf("常量: C1 %.1f  CR %.1f  BASE %.1f ms/token(= %.1f t/s)\n", C1, CR, BASE, 1000.0 / BASE);
    for (int i = 1; i < argc; i++) {
        text x = load(argv[i]);
        if (x.np < 3 || x.n <= x.np) { fprintf(stderr, "%s: 提示 %d / 总长 %d 不成形, 跳过\n", x.name, x.np, x.n); free(x.t); continue; }
        int ng = x.n - x.np; gen_total += ng;
        hit *HA = malloc((size_t)ng * sizeof *HA), *HB = malloc((size_t)ng * sizeof *HB);
        for (int q = x.np; q < x.n; q++) {
            HA[q - x.np] = best_match(&x, q, x.np - 1 < q - 2 ? x.np - 1 : q - 2);
            HB[q - x.np] = best_match(&x, q, q - 2);
            for (int v = 0; v < 2; v++) { const hit *h = v ? &HB[q - x.np] : &HA[q - x.np]; if (!h->m || !h->k) continue;
                int b = h->m < 6 ? 0 : h->m < 12 ? 1 : h->m < 24 ? 2 : 3; mb_prop[v][b]++; mb_ok[v][b] += h->acc > 0; }
        }
        stat fa = { 0 }, fb = { 0 }; walk(&x, HA, 3, 5, &fa); walk(&x, HB, 3, 5, &fb);
        printf("%-40s 提示 %5d 续写 %4d | A L3K5: ", x.name, x.np, ng); row("", &fa); printf("%-67s| B L3K5: ", ""); row("", &fb);
        for (int v = 0; v < 2; v++) for (int l = 0; l < NL; l++) for (int k = 0; k < NK; k++) walk(&x, v ? HB : HA, LS[l], KS[k], &agg[v][l][k]);
        free(HA); free(HB); free(x.t);
    }
    printf("\n合计 %ld 个生成 token\n", gen_total);
    for (int v = 0; v < 2; v++) {
        printf("%s 匹配长度桶(3~5 / 6~11 / 12~23 / 24+): 提案数 %ld %ld %ld %ld, 首位命中 %.1f%% %.1f%% %.1f%% %.1f%%\n", v ? "B 提示+前文" : "A 只提示",
               mb_prop[v][0], mb_prop[v][1], mb_prop[v][2], mb_prop[v][3],
               mb_prop[v][0] ? 100.0 * mb_ok[v][0] / mb_prop[v][0] : 0, mb_prop[v][1] ? 100.0 * mb_ok[v][1] / mb_prop[v][1] : 0,
               mb_prop[v][2] ? 100.0 * mb_ok[v][2] / mb_prop[v][2] : 0, mb_prop[v][3] ? 100.0 * mb_ok[v][3] / mb_prop[v][3] : 0);
        for (int l = 0; l < NL; l++) for (int k = 0; k < NK; k++) { char tag[16]; snprintf(tag, sizeof tag, "%s L%-2d K%d", v ? "B" : "A", LS[l], KS[k]); row(tag, &agg[v][l][k]); }
    }
    return 0;
}
