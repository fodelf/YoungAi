/* rplan_solve.c — 体积预算 → 逐层档位配置(2026-08-27, 用户令"按 86G 生成动态配置 json")。
 *
 * 【这个程序解决什么】平权量化给 43 层每层同样的位宽。但实测每层的量化误差差 5 倍
 * (vq86h_noz/plan.txt: L0 relh=0.1055, L28 relh=0.5315)。把易层省下的字节挪给难层,
 * 总体积不变而总误差更小 —— 前提是"加位宽真能减误差", 这由档梯 cos 表(实测)决定。
 *
 * 【为什么重写 rplan_solve_v4.py】① 全仓零 Python 裁决; ② 原版档梯只有 vq4x512 往【下】
 * 的档(cos 0.740/0.646/0.573/0.470), 所以在 86G 预算下它的最优解必然是"每层都拉满到
 * vq4x512"= 平权 —— 分配器根本没有往上的档可选。这就是当年 86G 上平权赢的机制原因,
 * 不是平权比动态好, 是动态在那个档梯下无处可去。本版档梯双向延伸。
 *
 * 【与 v4 的另一处不同: 不做动态专家】v4 把每层分成 hot(固定 vq4x512)/cold(便宜档)两档。
 * 本版一律 hot=0, 全 256 专家在层内同档 —— 因为"丢专家/降级专家"已被实测两次否决
 * (纯剪枝 q2 输出崩成乱码; 冷热分档在案裁决为设计缺陷)。所以分配变量只剩【每层档位】,
 * 即动态层 + 层内平权。rplan 的 hot=0 语义正好是"全专家走 dim/nc 档", 量化器零改动。
 *
 * 【难度权重为什么取平方差】d[L] = relh²[L] − relh²[L−1]。relh 是【累积】相对误差,
 * 直接用它会让 L0 虚高(L0 没有上游, 它的 relh 是从零开始的全部误差, 别的层是增量)。
 * 平方 = 误差能量, 才是可加的同一把尺。v4 原版踩过这个坑并修正, 此处沿用其结论。
 *
 * 【贪心为什么正确】每步挑"每字节换回的加权精度增益"最大的一层升档, 直到预算用完。
 * 档梯单调(越贵越准)时这是背包问题的标准贪心近似; 档梯非单调会制造负收益台阶把所有层
 * 卡死在最稀档(v4 原版实测事故), 所以启动时强制校验单调性。
 *
 * 用法: rplan_solve --held <plan.txt> --ladder <ladder.txt> --budget-gib N
 *                   [--out-json f.json] [--out-rplan f.txt] [--floor-nc N]
 *   plan.txt   量化器产的逐层账: "L=<n> cand=... relh=<v> gate=..."(取 L 与 relh)
 *   ladder.txt 每行 "dim nc cos  # 出处", 按字节从小到大; cos = 该档实测重建余弦
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vq_qc_bytes.h"

#define MAXL   64
#define MAXT   16
#define NEXP   256
#define MOEI   2048
#define DMODEL 4096

typedef struct { int dim, nc; double cos_; size_t bytes; char note[128]; } tier_t;

static size_t layer_bytes(const tier_t *t) {
    /* 与量化器 vq_total_bytes(hot=0 分支)逐字对应: 表头 + 256 专家 ×(w1+w3+w2)。
     * w2 与 w1/w3 同档(rplan 的 w2dim/w2nc 我们填成一样) —— 口径差一点, 体积账就对不上, 跑完
     * 5 小时才发现超预算。 */
    size_t hdr = 16 + (size_t)NEXP * 3 * 8;
    size_t w13 = vq_payload_bytes(MOEI, DMODEL, t->dim, t->nc);
    size_t w2  = vq_payload_bytes(DMODEL, MOEI, t->dim, t->nc);
    return hdr + (size_t)NEXP * (2 * w13 + w2);
}

static int parse_kv(const char *ln, const char *key, double *out) {
    const char *p = strstr(ln, key);
    if (!p) return 0;
    return sscanf(p + strlen(key), "%lf", out) == 1;
}

int main(int argc, char **argv) {
    const char *held_p = NULL, *lad_p = NULL, *oj = NULL, *ot = NULL;
    double budget_gib = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--held") && i + 1 < argc) held_p = argv[++i];
        else if (!strcmp(argv[i], "--ladder") && i + 1 < argc) lad_p = argv[++i];
        else if (!strcmp(argv[i], "--budget-gib") && i + 1 < argc) budget_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out-json") && i + 1 < argc) oj = argv[++i];
        else if (!strcmp(argv[i], "--out-rplan") && i + 1 < argc) ot = argv[++i];
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if (!held_p || !lad_p || budget_gib <= 0) {
        fprintf(stderr, "用法: rplan_solve --held <plan.txt> --ladder <ladder.txt> --budget-gib N\n"
                        "                 [--out-json f.json] [--out-rplan f.txt]\n");
        return 2;
    }

    /* ---- 档梯 ---- */
    tier_t T[MAXT]; int NT = 0;
    FILE *f = fopen(lad_p, "r");
    if (!f) { fprintf(stderr, "档梯打不开 %s\n", lad_p); return 2; }
    char ln[512];
    while (fgets(ln, sizeof ln, f)) {
        if (ln[0] == '#' || ln[0] == '\n') continue;
        tier_t t; memset(&t, 0, sizeof t);
        const char *h = strchr(ln, '#');
        if (sscanf(ln, "%d %d %lf", &t.dim, &t.nc, &t.cos_) != 3) continue;
        if (h) { snprintf(t.note, sizeof t.note, "%s", h + 1); char *nl = strchr(t.note, '\n'); if (nl) *nl = 0; }
        t.bytes = layer_bytes(&t);
        if (NT >= MAXT) { fprintf(stderr, "档梯过长\n"); return 2; }
        T[NT++] = t;
    }
    fclose(f);
    if (NT < 2) { fprintf(stderr, "★档梯至少两档, 只读到 %d★\n", NT); return 2; }
    /* 单调性硬校验: 越贵必须越准。非单调会在贪心里造负收益台阶, 把所有层卡死在最稀档
     * (v4 原版实测事故: v24x256 cos 0.468 < v32 的 0.470 却更贵, 全线停在 v32 升不上去)。 */
    for (int i = 1; i < NT; i++)
        if (T[i].bytes <= T[i-1].bytes || T[i].cos_ <= T[i-1].cos_) {
            fprintf(stderr, "★档梯非单调: 第%d档 vq%dx%d(%zuB cos%.4f) vs 第%d档 vq%dx%d(%zuB cos%.4f)"
                    " — 越贵必须越准, 否则贪心必卡死★\n",
                    i-1, T[i-1].dim, T[i-1].nc, T[i-1].bytes, T[i-1].cos_,
                    i, T[i].dim, T[i].nc, T[i].bytes, T[i].cos_);
            return 3;
        }

    /* ---- 逐层难度 ---- */
    double relh[MAXL]; int NL = 0;
    memset(relh, 0, sizeof relh);
    f = fopen(held_p, "r");
    if (!f) { fprintf(stderr, "held 表打不开 %s\n", held_p); return 2; }
    while (fgets(ln, sizeof ln, f)) {
        double L, v;
        if (!parse_kv(ln, "L=", &L) || !parse_kv(ln, "relh=", &v)) continue;
        int li = (int)L;
        if (li < 0 || li >= MAXL) continue;
        relh[li] = v; if (li + 1 > NL) NL = li + 1;
    }
    fclose(f);
    if (NL < 2) { fprintf(stderr, "★held 表只解析到 %d 层★\n", NL); return 2; }

    double w[MAXL];
    const double FLOOR = 0.004;   /* 自愈层(能量负增量)仍要参与竞价, 不能给 0 权重 */
    for (int L = 0; L < NL; L++) {
        double h2 = relh[L] * relh[L], p2 = L ? relh[L-1] * relh[L-1] : 0.0;
        w[L] = h2 - p2; if (w[L] < FLOOR) w[L] = FLOOR;
    }

    /* ---- 贪心分配 ---- */
    size_t B = (size_t)(budget_gib * (double)(1ull << 30));
    int tier[MAXL];
    size_t used = 0;
    for (int L = 0; L < NL; L++) { tier[L] = 0; used += T[0].bytes; }
    if (used > B) {
        fprintf(stderr, "★最稀档 %d 层已占 %.3f GiB > 预算 %.3f GiB★\n",
                NL, used / (double)(1ull << 30), budget_gib);
        return 3;
    }
    printf("[起点] %d 层全最稀档 vq%dx%d = %.3f GiB / 预算 %.3f GiB\n",
           NL, T[0].dim, T[0].nc, used / (double)(1ull << 30), budget_gib);

    long steps = 0;
    for (;;) {
        int bestL = -1; double bestR = 0.0;
        for (int L = 0; L < NL; L++) {
            if (tier[L] + 1 >= NT) continue;
            size_t dB = T[tier[L]+1].bytes - T[tier[L]].bytes;
            if (used + dB > B) continue;
            double r = w[L] * (T[tier[L]+1].cos_ - T[tier[L]].cos_) / (double)dB;
            if (r > bestR) { bestR = r; bestL = L; }
        }
        if (bestL < 0) break;
        used += T[tier[bestL]+1].bytes - T[tier[bestL]].bytes;
        tier[bestL]++; steps++;
    }
    double obj = 0, wsum = 0;
    for (int L = 0; L < NL; L++) { obj += w[L] * T[tier[L]].cos_; wsum += w[L]; }
    printf("[收敛] %ld 步升档, 落位 %.4f GiB / 预算 %.4f GiB (余 %.1f MiB), 加权精度 %.6f\n",
           steps, used / (double)(1ull << 30), budget_gib,
           (B - used) / 1048576.0, obj / wsum);

    /* ---- 逐层表 ---- */
    printf("\n%-4s %8s %9s %11s %9s %10s\n", "层", "relh", "难度w", "档位", "cos", "MiB");
    int hist[MAXT]; memset(hist, 0, sizeof hist);
    for (int L = 0; L < NL; L++) {
        hist[tier[L]]++;
        printf("L%-3d %8.4f %9.5f  vq%d x%-5d %9.4f %10.1f\n", L, relh[L], w[L],
               T[tier[L]].dim, T[tier[L]].nc, T[tier[L]].cos_, T[tier[L]].bytes / 1048576.0);
    }
    printf("\n档位分布: ");
    for (int t = 0; t < NT; t++) if (hist[t]) printf("vq%dx%d×%d  ", T[t].dim, T[t].nc, hist[t]);
    printf("\n");

    /* ---- 产物 ---- */
    if (ot) {
        FILE *o = fopen(ot, "w");
        if (!o) { fprintf(stderr, "写不了 %s\n", ot); return 4; }
        /* hot=0 = 全 256 专家走 dim/nc 档(层内平权); w2 与 w1w3 同档, 与体积账口径一致。 */
        for (int L = 0; L < NL; L++)
            fprintf(o, "L=%d dim=%d nc=%d hot=0 w2dim=%d w2nc=%d\n",
                    L, T[tier[L]].dim, T[tier[L]].nc, T[tier[L]].dim, T[tier[L]].nc);
        fclose(o);
        printf("计划表 → %s\n", ot);
    }
    if (oj) {
        FILE *o = fopen(oj, "w");
        if (!o) { fprintf(stderr, "写不了 %s\n", oj); return 4; }
        fprintf(o, "{\n  \"budget_GiB\": %.4f,\n  \"payload_GiB\": %.4f,\n"
                   "  \"objective\": %.6f,\n  \"n_layers\": %d,\n  \"hot\": 0,\n"
                   "  \"note\": \"hot=0 = 全256专家层内同档(动态层, 非动态专家)\",\n  \"ladder\": [\n",
                budget_gib, used / (double)(1ull << 30), obj / wsum, NL);
        for (int t = 0; t < NT; t++)
            fprintf(o, "    {\"dim\": %d, \"nc\": %d, \"cos\": %.4f, \"MiB\": %.1f, \"bpw\": %.4f}%s\n",
                    T[t].dim, T[t].nc, T[t].cos_, T[t].bytes / 1048576.0,
                    (double)(T[t].bytes - 16 - NEXP*3*8) * 8.0 / ((double)NEXP * 3 * MOEI * DMODEL),
                    t + 1 < NT ? "," : "");
        fprintf(o, "  ],\n  \"layers\": [\n");
        for (int L = 0; L < NL; L++)
            fprintf(o, "    {\"L\": %d, \"relh\": %.4f, \"weight\": %.5f, \"dim\": %d, \"nc\": %d,"
                       " \"cos\": %.4f, \"MiB\": %.1f}%s\n",
                    L, relh[L], w[L], T[tier[L]].dim, T[tier[L]].nc, T[tier[L]].cos_,
                    T[tier[L]].bytes / 1048576.0, L + 1 < NL ? "," : "");
        fprintf(o, "  ]\n}\n");
        fclose(o);
        printf("配置 JSON → %s\n", oj);
    }
    return 0;
}
