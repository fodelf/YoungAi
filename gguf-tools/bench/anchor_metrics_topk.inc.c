/* anchor_metrics_topk.inc.c — 后训练的主尺: 决策点上 argmax 到底选了谁(2026-09-13)。
 * 被 anchor_metrics.c 单 TU include(拆文件只为守 500 行), 用法:
 *     anchor_metrics --ref-topk topk.bin [--rows a:b,...] [--alt alt_ids.txt]
 *
 * 为什么需要这把尺: 部署是贪心 argmax。V4 那轮后训练把"对版−错版"的 NLL 差压掉 61.9%,
 * 决策探针的三个数字却★一个小数点都没动★ —— 因为判据仍 >0, 分歧位置上错版 token 的概率
 * (0.291)还是高于对版(0.219), argmax 当然照选错版。NLL 是连续量, 行为是离散的, 只有这个
 * 命中率能回答"之前的错误判断撬动了没有"。
 *
 * 输入是引擎 --score-topk 的产物, 读取走 src/common/ds4_etgd.h(引擎写、解算器读、这里读,
 * 全仓同一份)。tgt 已经是"位置 i 该预测的下一个 token"(引擎写的时候就错开一格了), 这里不再
 * 做任何偏移 —— 偏移只该有一处。
 * --alt 给另一版(错版)的 ids 文件时多出一组数: 在两版真正不同的那些位置上, argmax 站在
 * 目标一侧的比例, 以及 p(目标)−p(另一版) 的均值 —— 后者是"离翻转还有多远"。 */
#include "../../src/common/ds4_etgd.h"

/* alt 文件 = 另一版的完整 token 序列(每行一个 id), 行号与 topk 文件同一口径。 */
static int *alt_read(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "--alt %s 打不开\n", path); exit(2); }
    int cap = 8192, n = 0, v;
    int *a = malloc((size_t)cap * 4);
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) { cap *= 2; a = realloc(a, (size_t)cap * 4); }
        a[n++] = v;
    }
    fclose(f);
    *n_out = n;
    return a;
}

/* ★自检 2(back.md §4.5)★: 解算器预测的 logit 差 vs 真前向的。predict.txt 是候选目录里那张表,
 * 每行 "段 样本 行号 a b m0 dm m_pred"。只取属于本样本(smp)的行, 用【挂了候选之后】重打的
 * top-K 表算真 m = ln p(a) − ln p(b), 与 m_pred 对。
 * 为什么这针必须有: 预测与真前向对不上, 说明取料行号/α/inv/W 行/bf16 五处之一错位 —— 那是 bug,
 * 不是"方法不行"。09-13 第一版没有它, 0.20% 那个读数当时判不了是形态问题还是管子漏。 */
static int predict_check(const ds4_etgd *t, const char *pfile, int smp) {
    FILE *f = fopen(pfile, "r");
    if (!f) { fprintf(stderr, "--predict %s 打不开\n", pfile); return 2; }
    char seg[32], line[512];
    int n = 0, agree = 0, ndec = 0, dec_hit = 0;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, serr = 0;
    while (fgets(line, sizeof line, f)) {
        int sm, row, a, b;
        double m0, dm, mp;
        if (line[0] == '#') continue;
        if (sscanf(line, "%31s %d %d %d %d %lf %lf %lf", seg, &sm, &row, &a, &b, &m0, &dm, &mp) != 8) continue;
        if (sm != smp || row < 0 || row >= t->n) continue;
        const float pa = ds4_etgd_p_of(t, row, a), pb = ds4_etgd_p_of(t, row, b);
        if (!(pa > 0.f) || !(pb > 0.f)) continue;   /* 有一个掉出 top-K: 真 m 只有界没有值, 不进相关 */
        const double mr = log((double)pa) - log((double)pb);
        sx += mp; sy += mr; sxx += mp * mp; syy += mr * mr; sxy += mp * mr; serr += fabs(mr - mp);
        n++;
        if ((mr > 0) == (mp > 0)) agree++;
        if (strncmp(seg, "dec", 3) == 0) { ndec++; if (mr > 0) dec_hit++; }
    }
    fclose(f);
    if (n < 2) { printf("★自检 2: 本样本可比的行只有 %d 个, 不出数★\n", n); return 0; }
    const double cov = sxy / n - (sx / n) * (sy / n);
    const double sd1 = sqrt(sxx / n - (sx / n) * (sx / n)), sd2 = sqrt(syy / n - (sy / n) * (sy / n));
    printf("★自检2 预测 vs 真前向[样本 %d]: %d 行, 相关 %.4f, 翻/不翻一致 %.2f%%, 平均 |真−预测| %.4f★\n",
           smp, n, (sd1 > 0 && sd2 > 0) ? cov / (sd1 * sd2) : 0.0, 100.0 * agree / n, serr / n);
    if (ndec) printf("  其中决策点 %d 个: 真前向 m>0 的 %d 个 = %.2f%%\n", ndec, dec_hit, 100.0 * dec_hit / ndec);
    return 0;
}

/* ★"别处别动"的正确量法★(2026-09-13 夜, 第四个口径错改出来的): 比【两张 top-K 表】——
 * 基线态一张、挂了候选一张 —— 在同一批行上 argmax 是不是同一个 token。
 * 之前拿 --ref-topk 的"argmax 命中目标"当这把尺, 报的是"argmax 是否等于教师强制的下一个 token":
 * 那是【基线自身的属性】, 约束行本来就只有 62% 命中(报告段的普通位置, 模型最想说的词未必就是
 * 实际写出来的那个)。挂候选前 62.70%、挂后 62.64% —— 两次几乎一样, 恰恰说明它量的不是改动。 */
static int topk_vs(const char *pa, const char *pb, const char *rowspec) {
    ds4_etgd a, b;
    if (ds4_etgd_read(pa, &a)) return 2;
    if (ds4_etgd_read(pb, &b)) { ds4_etgd_free(&a); return 2; }
    if (a.n != b.n) { fprintf(stderr, "两张表行数不同(%d vs %d) —— 不是同一条序列\n", a.n, b.n); ds4_etgd_free(&a); ds4_etgd_free(&b); return 2; }
    int rn = a.n; int *rr = rows_range(0, a.n); const char *nm = "全段";
    if (rowspec) {
        free(rr); rr = rows_parse(rowspec, a.S, &rn);
        if (!rr) { fprintf(stderr, "--rows 解析失败\n"); ds4_etgd_free(&a); ds4_etgd_free(&b); return 2; }
        nm = "行段";
    }
    long same = 0, used = 0;
    double dp = 0.0;
    for (int q = 0; q < rn; q++) {
        const int i = rr[q];
        if (i < 0 || i >= a.n) continue;
        float p1a = 0.f, p1b = 0.f;
        const int ta = ds4_etgd_top1(&a, i, &p1a), tb = ds4_etgd_top1(&b, i, &p1b);
        if (ta < 0 || tb < 0) continue;
        used++;
        if (ta == tb) same++;
        dp += fabs((double)p1a - (double)ds4_etgd_p_of(&b, i, ta));   /* 基线 top1 的概率被动了多少 */
    }
    printf("★两态对比[%s]: argmax 没变 %ld/%ld = %.2f%%; 基线 top1 的概率平均动了 %.4f★\n",
           nm, same, used, used ? 100.0 * (double)same / (double)used : 0.0, used ? dp / (double)used : 0.0);
    free(rr); ds4_etgd_free(&a); ds4_etgd_free(&b);
    return 0;
}

static int topk_report(const char *path, const char *rowspec, const char *altp) {
    ds4_etgd t;
    if (ds4_etgd_read(path, &t)) return 2;
    int na = 0; int *alt = altp ? alt_read(altp, &na) : NULL;
    int rn = t.n; int *rr = rows_range(0, t.n); const char *nm = "全段";
    if (rowspec) {
        free(rr); rr = rows_parse(rowspec, t.S, &rn);
        if (!rr) { fprintf(stderr, "--rows 要 a:b 或 a:b,c:d,...(得到 \"%s\", S=%d)\n", rowspec, t.S); ds4_etgd_free(&t); return 2; }
        nm = "行段";
    }
    long hit = 0, used = 0, dec_hit = 0, dec_used = 0;
    double sp = 0, s1 = 0, sm = 0, sdiff = 0;
    for (int q = 0; q < rn; q++) {
        const int i = rr[q];
        if (i < 0 || i >= t.n || t.tgt[i] < 0) continue;   /* 末位没有下一个 token, 不判 */
        float p1 = 0.f;
        const int top1 = ds4_etgd_top1(&t, i, &p1);
        used++;
        if (top1 == t.tgt[i]) hit++;
        sp += t.tgt_p[i]; s1 += p1; sm += t.mass[i];
        if (alt && i + 1 < na) {
            const int aw = alt[i + 1];
            if (aw == t.tgt[i]) continue;                  /* 两版这里本来就一样, 不是决策点 */
            dec_used++;
            if (top1 == t.tgt[i]) dec_hit++;
            sdiff += (double)t.tgt_p[i] - (double)ds4_etgd_p_of(&t, i, aw);
        }
    }
    printf("★argmax 命中目标[%s]: %ld/%ld = %.2f%%★\n", nm, hit, used,
           used ? 100.0 * (double)hit / (double)used : 0.0);
    printf("  平均 p(目标) %.4f | 平均 p(top1) %.4f | 平均 top-K 覆盖质量 %.4f\n",
           used ? sp / (double)used : 0.0, used ? s1 / (double)used : 0.0, used ? sm / (double)used : 0.0);
    if (alt)
        printf("★决策点(两版不同的位置) %ld 个: argmax 已站到目标一侧 %ld 个 = %.2f%%; "
               "平均 p(目标)−p(另一版) = %+.4f★\n",
               dec_used, dec_hit, dec_used ? 100.0 * (double)dec_hit / (double)dec_used : 0.0,
               dec_used ? sdiff / (double)dec_used : 0.0);
    free(rr); free(alt); ds4_etgd_free(&t);
    return 0;
}

static int topk_predict(const char *path, const char *pfile, int smp) {
    ds4_etgd t;
    if (ds4_etgd_read(path, &t)) return 2;
    const int rc = predict_check(&t, pfile, smp);
    ds4_etgd_free(&t);
    return rc;
}
