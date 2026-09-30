/* posttrain_reward.c — 后训练 ③ 的奖励器(2026-09-29, back.md §14.5; 用户定: "根据复盘来, 预测对就是奖励, 不对就是惩罚")。
 * 对错只由代码按真实价格判, 不读任何 LLM 写的复盘文字。一份样本一行输出, 判不了的显式 INVALID(不当 0 分:
 * 否则 ③ 会学到"写坏 JSON = 安全")。
 *
 * 用法:
 *   posttrain_reward stock  <报告正文.txt> <日线.txt> <报告日 YYYYMMDD>
 *       日线.txt 每行 "YYYY-MM-DD open high low close"(qtf_capture_request.py ohlc 的输出), 次日 = 报告日之后第一根。
 *       输出: "R_pnl R_bin trade entry target stop rr next_date hit  R_pnl_t1 R_bin_t1"
 *         交易与否: 只认报告自己结尾 JSON 的三个价格, 按 CFO 4.4 的公式重算风报比 rr=(目标−入场)/(入场−止损),
 *                   rr ≥ 3(CFO 自己的铁律"≥3:1 才交易")且 目标>入场>止损 才算交易; 否则 R=0(不交易不奖不罚)。
 *         成交: 入场价在 [次日最低, 次日最高] 之外 = 没成交, R=0。
 *         成交后(§13.2): 最高≥目标 且 最低≤止损 → 按坏的算 (止损−入场)/入场, bin −1(同日先后不可知);
 *                        最高≥目标 → (目标−入场)/入场, bin +1;  最低≤止损 → (止损−入场)/入场, bin −1;
 *                        都没碰 → (收盘−入场)/入场, bin 0。  成交的再减双边成本 0.2%。
 *         第二列 *_t1: A 股 T+1 交收口径 —— 次日成交、次次日按同规则出场(次次日没有 = na), 只报不判。
 *   posttrain_reward market <正文.txt> <真值 上涨|下跌>
 *       正文里第一处不是在列选项("上涨"还是"下跌")的 "大盘上涨/大盘下跌" = 模型结论; 与真值同 → +1, 反 → −1。
 *       输出: "R_bin pred"。正文里没有结论 → INVALID。
 *   自测: cc -DPOSTTRAIN_REWARD_TEST posttrain_reward.c -o t && ./t(合成夹具, 答案手算)。 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PR_COST 0.002   /* 双边成本 0.2%(§13.2) */
#define PR_RR_MIN 3.0   /* CFO 提示词自己的铁律: 风险回报比 ≥ 3:1 才交易 */

typedef struct { char date[11]; double o, h, l, c; } pr_bar;

#ifndef POSTTRAIN_REWARD_TEST
static char *pr_read(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "★打不开 %s★\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f); b[n] = 0; if (len) *len = (size_t)n;
    return b;
}
#endif

/* 报告尾 JSON 里取一个数值键: 键名两侧引号单双都认(模型两种都写过), 值取第一个数字。找最后一次出现 ——
 * 报告正文里也可能提到 target_price 这个词, 结尾那份总结才是判决用的。返回 0 = 没找到。 */
static int pr_json_num(const char *text, const char *key, double *out) {
    const char *best = NULL, *p = text;
    const size_t kl = strlen(key);
    while ((p = strstr(p, key)) != NULL) {
        const char *q = p + kl;
        if (p > text && (p[-1] == '\'' || p[-1] == '"')) {
            if (*q == '\'' || *q == '"') q++;
            while (*q == ' ' || *q == '\t') q++;
            if (*q == ':') best = q + 1;
        }
        p += kl;
    }
    if (!best) return 0;
    while (*best == ' ' || *best == '\t' || *best == '\'' || *best == '"') best++;
    char *end = NULL;
    const double v = strtod(best, &end);
    if (end == best) return 0;
    *out = v;
    return 1;
}

#ifndef POSTTRAIN_REWARD_TEST
static int pr_bars_read(const char *path, pr_bar **out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "★打不开 %s★\n", path); return -1; }
    int cap = 64, n = 0;
    pr_bar *b = malloc((size_t)cap * sizeof *b);
    char line[256];
    while (fgets(line, sizeof line, f)) {
        pr_bar t;
        if (sscanf(line, "%10s %lf %lf %lf %lf", t.date, &t.o, &t.h, &t.l, &t.c) != 5) continue;
        if (n == cap) { cap *= 2; b = realloc(b, (size_t)cap * sizeof *b); }
        b[n++] = t;
    }
    fclose(f);
    *out = b;
    return n;
}
#endif

/* "YYYYMMDD" vs "YYYY-MM-DD" 的先后比较: 去掉横线按字符串比 */
static int pr_datecmp(const char *iso, const char *ymd) {
    char a[9] = {0}; int k = 0;
    for (const char *p = iso; *p && k < 8; p++) if (*p != '-') a[k++] = *p;
    return strcmp(a, ymd);
}

/* 一根日线上的出场结果: 返回 bin, *pnl = 相对入场的收益(未扣成本) */
static int pr_exit(const pr_bar *b, double entry, double target, double stop, double *pnl, const char **hit) {
    const int ht = b->h >= target, hs = b->l <= stop;
    if (ht && hs) { *pnl = (stop - entry) / entry; *hit = "both→stop"; return -1; }
    if (ht) { *pnl = (target - entry) / entry; *hit = "target"; return 1; }
    if (hs) { *pnl = (stop - entry) / entry; *hit = "stop"; return -1; }
    *pnl = (b->c - entry) / entry; *hit = "close"; return 0;
}

/* 个股奖励。返回 0 = 打出分; 2 = 报告判不了(INVALID); 3 = 还没有次日行情。 */
static int pr_stock(const char *report, const pr_bar *bars, int nb, const char *date, char *out, size_t cap) {
    double entry = 0, target = 0, stop = 0;
    if (!pr_json_num(report, "entry_price", &entry) || !pr_json_num(report, "target_price", &target) ||
        !pr_json_num(report, "stop_loss", &stop) || !(entry > 0) || !(target > 0) || !(stop > 0)) {
        snprintf(out, cap, "INVALID json(entry=%g target=%g stop=%g)", entry, target, stop); return 2; }
    int i1 = -1;
    for (int i = 0; i < nb; i++) if (pr_datecmp(bars[i].date, date) > 0) { i1 = i; break; }
    if (i1 < 0) { snprintf(out, cap, "INVALID no_next_bar(after %s)", date); return 3; }
    const double rr = (entry - stop) > 0 ? (target - entry) / (entry - stop) : 0.0;
    /* 1e-9 容差: 10.9/10.0/9.7 这种在十进制上正好 3.0 的组合, 二进制浮点算出来是 2.999…, 不能因此判成不交易 */
    const int trade = (rr + 1e-9 >= PR_RR_MIN) && target > entry && stop < entry;
    double r = 0, r1 = 0; int bin = 0, bin1 = 0; const char *hit = "no_trade", *hit1 = "na"; int has1 = 0;
    if (trade) {
        const pr_bar *b1 = &bars[i1];
        if (entry < b1->l || entry > b1->h) { hit = "no_fill"; }
        else { bin = pr_exit(b1, entry, target, stop, &r, &hit); r -= PR_COST; }
        if (i1 + 1 < nb) {   /* T+1 交收: 次日成交, 次次日出场 */
            has1 = 1;
            if (entry < b1->l || entry > b1->h) hit1 = "no_fill";
            else { bin1 = pr_exit(&bars[i1 + 1], entry, target, stop, &r1, &hit1); r1 -= PR_COST; }
        }
    }
    if (has1) snprintf(out, cap, "%.5f %d %d %.3f %.3f %.3f %.3f %s %s  %.5f %d", r, bin, trade, entry, target, stop, rr, bars[i1].date, hit, r1, bin1);
    else snprintf(out, cap, "%.5f %d %d %.3f %.3f %.3f %.3f %s %s  na na", r, bin, trade, entry, target, stop, rr, bars[i1].date, hit);
    return 0;
}

/* 大盘: 正文里第一处不是"列选项"的方向词。列选项 = 词后面紧跟 or/或/还是/斜杠, 或词前面紧挨着它们
 * (与 z_nightly_spark.sh reviewrun 段的 first() 同一套判法, 09-24 挂 ③ 的轨迹里实撞过 `… or "大盘上涨"?`)。 */
static int pr_listing(const char *text, const char *m, size_t mlen) {
    const char *q = m + mlen;
    while (*q == '"' || *q == ' ' || !strncmp(q, "”", 3) || !strncmp(q, "」", 3)) q += (*q == '"' || *q == ' ') ? 1 : 3;
    if (!strncmp(q, "or", 2) || !strncmp(q, "或", 3) || !strncmp(q, "还是", 6) || *q == '/') return 1;
    const char *p = m;
    while (p > text) {
        const char *r = p - 1;
        if (*r == '"' || *r == ' ') { p = r; continue; }
        if (r - 2 >= text && (!strncmp(r - 2, "“", 3) || !strncmp(r - 2, "「", 3))) { p = r - 2; continue; }
        break;
    }
    if (p - text >= 2 && !strncmp(p - 2, "or", 2)) return 1;
    if (p - text >= 3 && !strncmp(p - 3, "或", 3)) return 1;
    if (p - text >= 6 && !strncmp(p - 6, "还是", 6)) return 1;
    if (p > text && p[-1] == '/') return 1;
    return 0;
}

static const char *pr_market_pred(const char *text) {
    const char *keys[2] = { "大盘上涨", "大盘下跌" };
    const char *best = NULL; int bk = -1;
    for (int k = 0; k < 2; k++) {
        const char *p = text;
        while ((p = strstr(p, keys[k])) != NULL) {
            if (!pr_listing(text, p, strlen(keys[k]))) { if (!best || p < best) { best = p; bk = k; } break; }
            p += strlen(keys[k]);
        }
    }
    return bk < 0 ? NULL : (bk == 0 ? "上涨" : "下跌");
}

static int pr_market(const char *text, const char *truth, char *out, size_t cap) {
    const char *pred = pr_market_pred(text);
    if (!pred) { snprintf(out, cap, "INVALID no_conclusion"); return 2; }
    snprintf(out, cap, "%d %s", strcmp(pred, truth) == 0 ? 1 : -1, pred);
    return 0;
}

#ifndef POSTTRAIN_REWARD_TEST
int main(int argc, char **argv) {
    char out[512];
    if (argc == 5 && !strcmp(argv[1], "stock")) {
        char *rep = pr_read(argv[2], NULL); pr_bar *bars = NULL;
        const int nb = rep ? pr_bars_read(argv[3], &bars) : -1;
        if (!rep || nb < 0) return 1;
        const int rc = pr_stock(rep, bars, nb, argv[4], out, sizeof out);
        puts(out); free(rep); free(bars); return rc;
    }
    if (argc == 4 && !strcmp(argv[1], "market")) {
        char *txt = pr_read(argv[2], NULL);
        if (!txt) return 1;
        const int rc = pr_market(txt, argv[3], out, sizeof out);
        puts(out); free(txt); return rc;
    }
    fprintf(stderr, "用法: posttrain_reward stock <报告.txt> <日线.txt> <YYYYMMDD> | market <正文.txt> <上涨|下跌>\n");
    return 2;
}
#else
/* 合成夹具: 报告尾 JSON + 两根日线, 答案手算。 */
static int t_fail = 0;
#define T_EQ(cond, msg) do { if (!(cond)) { printf("FAIL %s\n", msg); t_fail++; } else printf("ok   %s\n", msg); } while (0)
int main(void) {
    char out[512];
    pr_bar bars[2] = { { "2026-09-04", 10.0, 11.0, 9.5, 10.5 }, { "2026-09-05", 10.4, 12.5, 10.3, 12.0 } };
    /* 1) 交易(rr=4): 入场 10, 目标 12, 止损 9.5 → 次日 high 11 < 12, low 9.5 ≤ 9.5 → 止损 −5% −0.2% */
    const char *r1 = "## 7. 总结\n{'symbol':'000001', 'entry_price': 10.0, 'target_price': 12.0,'stop_loss': 9.5, 'risk_return_ratio': 4.0}";
    T_EQ(pr_stock(r1, bars, 2, "20260903", out, sizeof out) == 0 && !strncmp(out, "-0.05200 -1 1 ", 14), "止损扫到 = 惩罚 (-0.052)");
    printf("   %s\n", out);
    /* 2) 同上但目标 10.8 ⇒ rr=1.6 < 3 ⇒ 不交易 ⇒ 0 */
    const char *r2 = "{\"entry_price\": 10.0, \"target_price\": 10.8, \"stop_loss\": 9.5}";
    T_EQ(pr_stock(r2, bars, 2, "20260903", out, sizeof out) == 0 && !strncmp(out, "0.00000 0 0 ", 12), "风报比 <3 = 不交易 = 0");
    /* 3) 目标被次日最高触到: 入场 10, 目标 10.9, 止损 9.7 ⇒ rr=3.0 ⇒ 交易; high 11 ≥ 10.9, low 9.5 ≤ 9.7 ⇒ 两个都碰 ⇒ 按坏的算 */
    const char *r3 = "{'entry_price': 10.0, 'target_price': 10.9, 'stop_loss': 9.7}";
    T_EQ(pr_stock(r3, bars, 2, "20260903", out, sizeof out) == 0 && !strncmp(out, "-0.03200 -1 1 ", 14), "目标与止损同日都碰 = 按坏的算");
    /* 4) 目标触到且止损没碰: 止损 9.4 ⇒ rr=(0.9)/(0.6)=1.5 <3 ⇒ 不交易; 改目标 11.0 止损 9.6: rr=1/0.4=2.5 不交易; 目标 11 止损 9.7: rr=3.33 → high 11 ≥ 11 ✓ low 9.5 ≤ 9.7 ✓ 两碰. 用第二根: 报告日 09-04 → 次日 09-05 high 12.5 low 10.3 */
    const char *r4 = "{'entry_price': 10.4, 'target_price': 12.0, 'stop_loss': 9.9}";   /* rr = 1.6/0.5 = 3.2; 09-05: 入场 10.4 ∈ [10.3,12.5], high ≥ 12 ✓, low 10.3 > 9.9 ✓ */
    T_EQ(pr_stock(r4, bars, 2, "20260904", out, sizeof out) == 0 && !strncmp(out, "0.15185 1 1 ", 12), "目标触到 = 奖励 (+0.1538 − 0.002)");
    printf("   %s\n", out);
    /* 5) 没成交: 入场 9.0 不在 [9.5, 11] */
    const char *r5 = "{'entry_price': 9.0, 'target_price': 12.0, 'stop_loss': 8.5}";   /* rr = 3/0.5 = 6 */
    T_EQ(pr_stock(r5, bars, 2, "20260903", out, sizeof out) == 0 && !strncmp(out, "0.00000 0 1 ", 12) && strstr(out, "no_fill"), "入场价不在次日区间 = 没成交 = 0");
    /* 6) JSON 缺字段 → INVALID */
    T_EQ(pr_stock("no json here", bars, 2, "20260903", out, sizeof out) == 2, "没有 JSON = INVALID");
    /* 7) 没有次日行情 → 3 */
    T_EQ(pr_stock(r1, bars, 2, "20260905", out, sizeof out) == 3, "报告日之后没有日线 = 等行情");
    /* 大盘 */
    T_EQ(pr_market("预测结果：大盘上涨\nJSON输出：{'prediction': '大盘上涨'}", "上涨", out, sizeof out) == 0 && !strncmp(out, "1 上涨", 7), "大盘方向对 = +1");
    T_EQ(pr_market("是 \"大盘上涨\" 还是 \"大盘下跌\"? 综合判断: 大盘下跌", "上涨", out, sizeof out) == 0 && !strncmp(out, "-1 下跌", 8), "跳过列选项后取第一处结论, 错 = -1");
    T_EQ(pr_market("I lean towards \"大盘下跌\" or \"大盘上涨\"? Hmm", "下跌", out, sizeof out) == 2, "只有列选项 = INVALID");
    printf(t_fail ? "★posttrain_reward 自测 %d 项失败★\n" : "posttrain_reward 自测全过\n", t_fail);
    return t_fail ? 1 : 0;
}
#endif
