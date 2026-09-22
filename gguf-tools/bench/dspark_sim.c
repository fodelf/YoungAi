/* dspark_sim.c — 投机调度器的离线陪审团(2026-09-19)。
 *
 * 干什么: 拿教师强制取料(每个位置的 5 位草稿 + 5 个 conf, core_v41_dcap.c 落的 <pairs.bin>.fix)与同一串真实 id,
 * 在主机上把"在线一轮"的账重放一遍: 任何调度策略 × 任何成本假设, 直接算出整段的 ms/token。
 * 为什么要它: 调度器的每个常量(草稿钱 / 每验一行的钱 / 判亏本歇几步)在 spark 上试一版就是 3 分钟一趟, 而且一趟只看得到
 * 它自己走过的位置 —— 歇着的那 58% 的步(fix4 实测)到底该不该歇, 在线永远量不到。取料每个位置都有草稿, 所以任何策略的轨迹
 * 都能精确重放: 温 0 + 同轨 ⇒ 验 k 位的接受数 = min(k, 草稿前缀与真序列一致的长度)。
 *
 * 用法: dspark_sim <pairs.bin.fix> <ids.txt> [graph_ms] [draft_ms] [v1_ms] [tok_ms] [cool]
 *   graph_ms = 纯解码走图一步(fix4: 37.97)   draft_ms = 草稿一轮(13.5)
 *   v1_ms    = 直发验证 1 行含同步/argmax(42.5)  tok_ms = 每多验一行(12.95)   cool = 现役冷却步数(16)
 * 出错会怎样: ids 与取料不是同一串 ⇒ 接受率算出来接近 0, 表里 p1 与引擎在线的 p1 对不上 —— 先核 ids 来源。
 * 编译: make -C gguf-tools dspark_sim */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAXB 8
typedef struct { char magic[4]; uint32_t n_target, d, block, n; } dfix_hdr;
typedef struct { float conf[MAXB]; int32_t draft[MAXB]; uint32_t acc; } pos_t;   /* acc = 与真序列一致的草稿前缀长 */

typedef struct { double graph, draft, v1, tok; } cost_t;   /* ms */
/* 调度器(core_draft_sched.c 同式): 以纯解码一步为 1 的比值; c_v1 = 直发验证 1 行 ÷ 走图一步(现役代码写死 c_fix + c_tok = 1) */
typedef struct { double c_draft, c_v1, c_tok; } sched_t;

static double sigm(float c) { return isfinite(c) ? 1.0 / (1.0 + exp(-(double)c)) : 0.0; }
static uint32_t pick_k(const sched_t *s, const float *conf, uint32_t B, double *val) {
    double surv = 1.0, sum = 0.0, best = 1.0 / (s->c_draft + s->c_v1); uint32_t bk = 0;
    for (uint32_t j = 0; j < B; j++) {
        surv *= sigm(conf[j]); sum += surv;
        const double v = (1.0 + sum) / (s->c_draft + s->c_v1 + s->c_tok * (double)(j + 1));
        if (v > best) { best = v; bk = j + 1; }
    }
    *val = best; return bk;
}
/* 重放: 位置 i = 刚吐出的 token 是 ids[i], 本步决定 ids[i+1]。pinned>0 = 钉死 k; oracle = 直接看真接受数(上限);
 * cool = 判亏本后歇几步(0 = 不歇, 每步都出草稿)。返回 ms/token。 */
static uint32_t g_start = 0;   /* 从第几对起重放(= 提示长度 - 1): 在线只在生成区出草稿, 提示区是人写的, 不算账 */
static double replay(const pos_t *P, uint32_t n, const cost_t *c, const sched_t *s, int pinned, int oracle, int cool,
                     double *rounds_out, double *rest_out, double *acc_out) {
    double ms = 0; uint32_t i = g_start < n ? g_start : 0, toks = 0, rounds = 0, rest = 0, acc = 0; int cd = 0;
    while (i < n) {
        if (pinned < 0) { ms += c->graph; i++; toks++; rest++; continue; }              /* 纯解码 */
        if (cd > 0) { cd--; ms += c->graph; i++; toks++; rest++; continue; }
        uint32_t k; double val = 2.0;
        if (oracle) { k = P[i].acc; if (k == 0) { ms += c->graph; i++; toks++; rest++; continue; } }
        else if (pinned > 0) k = (uint32_t)pinned;
        else k = pick_k(s, P[i].conf, MAXB, &val);
        const uint32_t a = k < P[i].acc ? k : P[i].acc;
        ms += c->draft + c->v1 + c->tok * (double)k;
        rounds++; acc += a;
        i += 1u + a; toks += 1u + a;
        if (!pinned && !oracle && val < 1.0 && cool > 0) cd = cool;
    }
    if (rounds_out) *rounds_out = rounds; if (rest_out) *rest_out = rest; if (acc_out) *acc_out = rounds ? (double)acc / rounds : 0;
    return ms / (double)toks;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: dspark_sim <pairs.bin.fix> <ids.txt> [graph_ms] [draft_ms] [v1_ms] [tok_ms] [cool]\n"); return 1; }
    cost_t c = { 37.97, 13.5, 42.5, 12.95 };
    if (argc > 3) c.graph = atof(argv[3]); if (argc > 4) c.draft = atof(argv[4]);
    if (argc > 5) c.v1 = atof(argv[5]);    if (argc > 6) c.tok = atof(argv[6]);
    const int cool = argc > 7 ? atoi(argv[7]) : 16;
    const uint32_t np = argc > 8 ? (uint32_t)atoi(argv[8]) : 0;   /* 提示 token 数: 重放与校准只看生成区 */
    g_start = np > 1u ? np - 1u : 0u;
    /* ids */
    FILE *fi = fopen(argv[2], "r"); if (!fi) { perror(argv[2]); return 1; }
    int32_t *ids = malloc(sizeof(int32_t) * 1u << 20); uint32_t nid = 0;
    while (nid < (1u << 20) && fscanf(fi, "%d", &ids[nid]) == 1) nid++;
    fclose(fi);
    /* fix */
    FILE *ff = fopen(argv[1], "rb"); if (!ff) { perror(argv[1]); return 1; }
    dfix_hdr h; if (fread(&h, sizeof h, 1, ff) != 1 || memcmp(h.magic, "DFIX", 4)) { fprintf(stderr, "不是 DFIX 文件\n"); return 1; }
    if (h.block > MAXB) { fprintf(stderr, "block %u > %d\n", h.block, MAXB); return 1; }
    pos_t *P = calloc(h.n, sizeof *P);
    const size_t skip = (size_t)h.n_target * h.d * 4;
    for (uint32_t p = 0; p < h.n; p++) {
        int32_t tok; float conf[MAXB]; int32_t dr[MAXB];
        if (fseek(ff, (long)skip, SEEK_CUR) || fread(&tok, 4, 1, ff) != 1 || fread(dr, 4, h.block, ff) != h.block ||
            fread(conf, 4, h.block, ff) != h.block) { fprintf(stderr, "第 %u 对截断\n", p); return 1; }
        const uint32_t i = 1u + p;                       /* 第 p 对 = 主模型位置 pos0 + p = 1 + p, 块首位 = ids[i] */
        if (i < nid && tok != ids[i]) { fprintf(stderr, "★第 %u 对块首位 %d ≠ ids[%u] %d: 取料与 ids 不是同一串★\n", p, tok, i, ids[i]); return 1; }
        uint32_t a = 0;
        while (a < h.block && i + 1u + a < nid && dr[a] == ids[i + 1u + a]) a++;
        P[p].acc = a; memcpy(P[p].conf, conf, sizeof conf); memcpy(P[p].draft, dr, sizeof dr);
    }
    fclose(ff);
    const uint32_t n = h.n;
    printf("[sim] %u 位置(块 %u), ids %u, 从第 %u 对(生成区)起算; 成本 ms: 走图一步 %.2f / 草稿 %.1f / 直发验 1 行 %.1f / 每多一行 %.2f; 冷却 %d 步\n",
           n, h.block, nid, g_start, c.graph, c.draft, c.v1, c.tok, cool);
    /* conf 校准: 第 j 位 sigmoid(conf_j) 分档 vs 条件接受 P(acc > j | acc ≥ j) */
    for (uint32_t j = 0; j < 3; j++) {
        uint32_t cnt[10] = { 0 }, hit[10] = { 0 };
        for (uint32_t p = g_start; p < n; p++) {
            if (P[p].acc < j) continue;
            int b = (int)(sigm(P[p].conf[j]) * 10.0); if (b > 9) b = 9; if (b < 0) b = 0;
            cnt[b]++; if (P[p].acc > j) hit[b]++;
        }
        printf("[校准] 第 %u 位 conf 档→条件接受:", j + 1);
        for (int b = 0; b < 10; b++) if (cnt[b]) printf("  %.1f:%.2f(%u)", b / 10.0, (double)hit[b] / cnt[b], cnt[b]);
        printf("\n");
    }
    {   /* 每位无条件/条件接受率 */
        uint32_t ge[MAXB + 1] = { 0 };
        for (uint32_t p = g_start; p < n; p++) for (uint32_t j = 0; j <= P[p].acc && j <= h.block; j++) ge[j]++;
        printf("[接受] 生成区全位置(%u): p1 %.3f", n - g_start, (double)ge[1] / (n - g_start));
        for (uint32_t j = 2; j <= h.block; j++) printf("  p%u|%u %.3f", j, j - 1, ge[j - 1] ? (double)ge[j] / ge[j - 1] : 0.0);
        printf("\n");
    }
    /* 策略 × 成本场景 */
    const sched_t old = { 0.285, 1.0, 0.385 };   /* 现役常量(c_fix + c_tok = 1 ⇒ c_v1 = 1) */
    const sched_t now = { c.draft / c.graph, c.v1 / c.graph, c.tok / c.graph };
    struct { const char *nm; cost_t c; } sc[4] = {
        { "实测", c },
        { "投机轮进图(验证基 = 走图一步, 草稿 -1.1)", { c.graph, c.draft - 1.1, c.graph, c.tok } },
        { "并集核(每行 ×0.65)", { c.graph, c.draft, c.v1, c.tok * 0.65 } },
        { "两者都做", { c.graph, c.draft - 1.1, c.graph, c.tok * 0.65 } },
    };
    for (int s = 0; s < 4; s++) {
        const cost_t *cc = &sc[s].c;
        const sched_t ns = { cc->draft / cc->graph, cc->v1 / cc->graph, cc->tok / cc->graph };
        printf("== 场景: %s\n", sc[s].nm);
        double r, rs, ac, m;
        m = replay(P, n, cc, &now, -1, 0, 0, &r, &rs, &ac);      printf("  纯解码                 %6.2f ms/token = %5.2f t/s\n", m, 1000 / m);
        m = replay(P, n, cc, &old, 0, 0, cool, &r, &rs, &ac);    printf("  现役常量+冷却 %-2d       %6.2f ms/token = %5.2f t/s  轮 %.0f 歇步 %.0f 均接受 %.2f\n", cool, m, 1000 / m, r, rs, ac);
        m = replay(P, n, cc, &ns, 0, 0, cool, &r, &rs, &ac);     printf("  重标常量+冷却 %-2d       %6.2f ms/token = %5.2f t/s  轮 %.0f 歇步 %.0f 均接受 %.2f\n", cool, m, 1000 / m, r, rs, ac);
        for (int cd = 0; cd <= 8; cd += 4) {
            m = replay(P, n, cc, &ns, 0, 0, cd, &r, &rs, &ac);   printf("  重标常量+冷却 %-2d       %6.2f ms/token = %5.2f t/s  轮 %.0f 歇步 %.0f 均接受 %.2f\n", cd, m, 1000 / m, r, rs, ac);
        }
        for (int k = 1; k <= (int)h.block; k++) {
            m = replay(P, n, cc, &ns, k, 0, 0, &r, &rs, &ac);    printf("  钉死 k=%d               %6.2f ms/token = %5.2f t/s  轮 %.0f 均接受 %.2f\n", k, m, 1000 / m, r, ac);
        }
        m = replay(P, n, cc, &ns, 0, 1, 0, &r, &rs, &ac);        printf("  先知(知道真接受数)     %6.2f ms/token = %5.2f t/s  轮 %.0f 歇步 %.0f 均接受 %.2f\n", m, 1000 / m, r, rs, ac);
    }
    free(P); free(ids);
    return 0;
}
