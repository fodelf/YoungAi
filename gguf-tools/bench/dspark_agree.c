/* dspark_agree.c — 把"投机接受率上限"的三个一致率一次量清楚(mtp-1.md M0′(c), 2026-09-17; 09-18 修尺)。
 *
 * 为什么必须是三个数而不是一个: 草稿塔(DSpark 三塔)是照**原始 FP 模型**训的, 出厂精度透传;
 * 而我们部署的主干是 VQ 1.5 bpw + 反修。于是引擎在线量到的"接受率"其实是两条偏离叠在一起:
 *     p1 ≈ (草稿器↔FP 原模型) × (底座↔FP 原模型)
 * 只量到一个 0.56 根本分不清该怪谁 —— 09-16 就是拿这一个数当依据去"把草稿器掰向底座", 判负两次。
 * 三个数一起摆出来才有下一步:
 *   ①草稿器↔底座 = 在线 p1 的上限(引擎 --dspark-capture 直接报)
 *   ②底座↔FP 锚  = 量化+反修还了多少原模型(= 五指标里的 Same top, 质量线的数)
 *   ③草稿器↔FP 锚 = **草稿器自己准不准**(它本来就该对着 FP 说话)
 * 读法:
 *   ③高(≈0.85)而①低 ⇒ 草稿器没毛病, 天花板就是②, 要抬接受率只能抬②(质量线)或让草稿器改盯底座;
 *   ③也低(≈0.6)     ⇒ 病在**我们这边的草稿器实现**(取 main_hidden 的位置/类型分发/块语义)或输入分布偏了,
 *                      那要金标夹具(FP main_hidden 喂三塔)再分一刀。
 *
 * ★09-18 修的两处尺子病(第一版两处都中, 09-17 那三个数 0.5707/0.6356/0.5082 全作废)★
 *   a) 取料错位: 引擎第一版把 ids[i] 在位置 i+1 又摆了一遍再出草稿(core_v41_dcap.c 文件头), ①③ 全量低;
 *      新取料头里带 pos0(第 0 对对应的主模型位置), 本工具按 pos0 + 对号 去对锚的行。
 *   b) 锚是**另一个模型**的: 唯一的 finj DQA2 锚是 V4 Flash 的(DIM 4096/43 层), 而 V4.1 是 5120 维/40 层;
 *      拿它算②③, 算出来的是"V4.1 底座 vs V4 原模型"。V4.1 的 FP 教师(v41_teacher.py)落的是
 *      <i32 S><i32 V><f32 logits> 裸格式, 本工具现在两种都吃; DQA2 的 DIM 与取料的 D 对不上直接停车。
 *
 * 输入: ①引擎 --dspark-capture 落的 pairs.bin(尾部存着每对的"底座 argmax"与"草稿首位"两串 token)
 *       ②FP 锚: DQA2(V4 caliper 格式)或 <S><V><logits>(V4.1 教师, 与 anchor_metrics --ref-raw 同一种)
 * 用法: dspark_agree <pairs.bin> <anchor.bin>
 * 编译: make -C gguf-tools dspark_agree
 *
 * ★口径对齐是这把尺的命门★: 第 p 对喂到主模型位置 pos0+p、预测 pos0+p+1; 锚的第 r 行是"位置 r 的 logits"
 * ⇒ 第 p 对配锚第 pos0+p 行。两边错一行就会得出一个看着合理其实全假的数, 所以 n 与 S、pos0 都打出来。 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { char magic[4]; uint32_t d, n, pos0; } dcap_hdr;

/* 打开锚, 把文件指针停在"位置 row0 的 logits 行"上; 返回词表大小与总行数 */
static FILE *open_anchor(const char *path, uint32_t cap_d, long long row0, long long *S_out, long long *V_out) {
    FILE *fa = fopen(path, "rb");
    if (!fa) { fprintf(stderr, "%s 打不开\n", path); exit(1); }
    uint32_t first = 0;
    if (fread(&first, 4, 1, fa) != 1) { fprintf(stderr, "%s 读不到头\n", path); exit(1); }
    long long S, V, base;
    if (first == 0x32415144u) {   /* 'DQA2' —— V4 caliper 锚: <8×u32 头><8B idh><fin/ridx/rw/H><logits> */
        uint32_t hd[8]; hd[0] = first;
        if (fread(hd + 1, 4, 7, fa) != 7) { fprintf(stderr, "%s: DQA2 头截断\n", path); exit(1); }
        const long long HCM = hd[2], DIM = hd[3], NL = hd[4], NACT = hd[6];
        S = hd[1]; V = hd[5];
        printf("[agree] 锚 %s: DQA2, S=%lld DIM=%lld NL=%lld VOCAB=%lld\n", path, S, DIM, NL, V);
        if ((uint32_t)DIM != cap_d) {
            fprintf(stderr, "★锚的隐维 %lld ≠ 取料的 %u: 这份锚是另一个模型的(09-17 实撞: V4 Flash 的锚配 V4.1 的取料),\n"
                            "  算出来的②③是两个模型之间的一致率, 不是量化损失。换 V4.1 教师的 <S><V><logits> 文件。★\n",
                    DIM, cap_d);
            exit(1);
        }
        base = 8 * 4 + 8 + NL * S * DIM * 4 + NL * S * NACT * 4 * 2 + NL * S * HCM * DIM * 4;
    } else {                      /* <i32 S><i32 V><f32 logits[S][V]>: v41_teacher.py / anchor_metrics --ref-raw 口径 */
        int32_t v = 0;
        if (fread(&v, 4, 1, fa) != 1) { fprintf(stderr, "%s: 头截断\n", path); exit(1); }
        S = (int32_t)first; V = v;
        if (S <= 0 || V < 1024 || V > (1 << 22)) { fprintf(stderr, "%s: 既不是 DQA2 也不像 <S><V><logits>(S=%lld V=%lld)\n", path, S, V); exit(1); }
        printf("[agree] 锚 %s: 裸 <S><V><logits>, S=%lld VOCAB=%lld\n", path, S, V);
        base = 8;
    }
    if (row0 >= S) { fprintf(stderr, "锚只有 %lld 行, 取料从位置 %lld 起, 对不上\n", S, row0); exit(1); }
    if (fseek(fa, (long)(base + row0 * V * 4), SEEK_SET) != 0) { fprintf(stderr, "锚定位失败\n"); exit(1); }
    *S_out = S; *V_out = V;
    return fa;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: dspark_agree <pairs.bin> <anchor.bin>\n"); return 1; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "%s 打不开\n", argv[1]); return 1; }
    dcap_hdr h;
    if (fread(&h, sizeof h, 1, fp) != 1 || memcmp(h.magic, "DCAP", 4)) { fprintf(stderr, "%s 不是取料文件\n", argv[1]); return 1; }
    const long long D = h.d, n = h.n, pos0 = h.pos0;
    if (pos0 == 0)
        printf("★取料头 pos0 = 0: 这是 09-18 修尺之前取的料, 草稿首位是错位的(见 core_v41_dcap.c 文件头), ①③不能信★\n");
    /* 两串 token 在文件尾: [主模型 argmax i32 × n][草稿首位 i32 × n] */
    if (fseek(fp, (long)(sizeof h + (long long)n * D * 2 * 4), SEEK_SET) != 0) { fprintf(stderr, "定位失败\n"); return 1; }
    int32_t *mtok = malloc((size_t)n * 4), *dtok = malloc((size_t)n * 4);
    if (!mtok || !dtok || fread(mtok, 4, (size_t)n, fp) != (size_t)n || fread(dtok, 4, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "token 串读不全(取料被截断?)\n"); return 1;
    }
    fclose(fp);

    /* ★顺手量一眼两边出口隐态的几何关系★(2026-09-17): 取料文件里每对都存着
     * X = 草稿器喂给出口头的隐态、Y = 主模型喂给同一个头的隐态。它们过的是**同一个头**,
     * 所以 argmax 差多少, 根子上就是这两个向量差多少。
     *   cos 高、范数比 ≈1 ⇒ 草稿器方向是对的, 差的是细节(它只是个小模型);
     *   cos 低或范数比离 1 很远 ⇒ 出口那一段有结构性问题(norm/缩放/取错了哪一层),
     *     那不是"模型弱", 是实现欠账 —— 而且这种情况下拿线性映射去对齐当然解不出来。
     * (09-17 错位料上量到 cos 0.32: 那是"猜 X X 后面"对"猜 X 后面"的两份隐态, 本来就不该像。) */
    {
        FILE *fx = fopen(argv[1], "rb");
        float *xb = (float *)malloc((size_t)D * 2 * 4);
        if (fx && xb && fseek(fx, sizeof h, SEEK_SET) == 0) {
            double csum = 0, rsum = 0; long long cn = 0;
            const long long probe = n < 512 ? n : 512;   /* 512 对够看形态, 不用读整份 */
            for (long long i = 0; i < probe; i++) {
                if (fread(xb, 4, (size_t)D * 2, fx) != (size_t)D * 2) break;
                double dot = 0, nx = 0, ny = 0;
                for (long long d = 0; d < D; d++) { dot += (double)xb[d] * xb[D + d]; nx += (double)xb[d] * xb[d]; ny += (double)xb[D + d] * xb[D + d]; }
                if (nx > 0 && ny > 0) { csum += dot / (sqrt(nx) * sqrt(ny)); rsum += sqrt(nx) / sqrt(ny); cn++; }
            }
            if (cn) printf("[agree] 出口隐态(前 %lld 对): cos(草稿, 主模型) = %.4f, 范数比 ‖草稿‖/‖主模型‖ = %.4f\n",
                           cn, csum / cn, rsum / cn);
        }
        if (fx) fclose(fx);
        free(xb);
    }

    long long S = 0, VOCAB = 0;
    FILE *fa = open_anchor(argv[2], (uint32_t)D, pos0, &S, &VOCAB);
    const long long rows = n < S - pos0 ? n : S - pos0;
    printf("[agree] 取料 %lld 对(第 0 对 = 位置 %lld) / 锚 %lld 行 ⇒ 逐行对 %lld 对(锚第 %lld..%lld 行)\n",
           n, pos0, S, rows, pos0, pos0 + rows - 1);
    float *row = malloc((size_t)VOCAB * 4);
    long long ab = 0, af = 0, df = 0;   /* 草稿↔底座 / 底座↔FP / 草稿↔FP */
    /* ★按"这个位置好不好猜"分档★(2026-09-17): 只看一个总的一致率分不出"草稿器有结构性缺陷"与
     * "它只是个弱模型"。用 FP 锚自己的 top1−top2 logit 间隔当难度: 间隔大 = 这个位置连小模型都该猜中。
     * 读法: 最容易那一档 ①还只有六成上下 ⇒ 结构性缺陷(实现错了或输入喂错);
     *       最容易那一档 ①接近九成、难档才掉 ⇒ 草稿器是对的, 只是弱, 天花板就在那儿。 */
    enum { NB = 4 };
    static const float cut[NB] = { 1.0f, 3.0f, 6.0f, 1e30f };   /* top1−top2 的 logit 间隔分档 */
    long long bn[NB] = {0}, bab[NB] = {0}, baf[NB] = {0}, bdf[NB] = {0};
    for (long long i = 0; i < rows; i++) {
        if (fread(row, 4, (size_t)VOCAB, fa) != (size_t)VOCAB) { fprintf(stderr, "锚第 %lld 行截断\n", pos0 + i); return 1; }
        long long best = 0;
        for (long long v = 1; v < VOCAB; v++) if (row[v] > row[best]) best = v;   /* 并列取小下标, 与引擎 argmax 同规矩 */
        float second = -1e30f;
        for (long long v = 0; v < VOCAB; v++) if (v != best && row[v] > second) second = row[v];
        const float margin = row[best] - second;
        int b = 0; while (b < NB - 1 && margin >= cut[b]) b++;
        bn[b]++;
        if (dtok[i] == mtok[i]) { ab++; bab[b]++; }
        if (mtok[i] == (int32_t)best) { af++; baf[b]++; }
        if (dtok[i] == (int32_t)best) { df++; bdf[b]++; }
    }
    printf("\n★按 FP 锚的 top1−top2 间隔分档(越靠后越好猜)★\n");
    printf("  %-14s %8s %10s %10s %10s\n", "间隔档", "位置数", "①草↔底", "②底↔FP", "③草↔FP");
    for (int b = 0; b < NB; b++) {
        if (!bn[b]) continue;
        char lab[32];
        if (b == 0) snprintf(lab, sizeof lab, "< %.0f", cut[0]);
        else if (b == NB - 1) snprintf(lab, sizeof lab, ">= %.0f", cut[NB - 2]);
        else snprintf(lab, sizeof lab, "%.0f ~ %.0f", cut[b - 1], cut[b]);
        printf("  %-14s %8lld %10.4f %10.4f %10.4f\n", lab, bn[b],
               (double)bab[b] / bn[b], (double)baf[b] / bn[b], (double)bdf[b] / bn[b]);
    }
    printf("\n★三个一致率(n=%lld)★\n", rows);
    printf("  ①草稿器 ↔ 部署底座 = %.4f   (= 在线首位接受率 p1 的上限)\n", (double)ab / (double)rows);
    printf("  ②部署底座 ↔ FP 锚  = %.4f   (= 量化+反修还了多少原模型; 该与五指标的 Same top 一致)\n", (double)af / (double)rows);
    printf("  ③草稿器 ↔ FP 锚    = %.4f   (= 草稿器自己准不准; 它本来就对着 FP 说话)\n", (double)df / (double)rows);
    printf("\n读法: ③≈0.85 而①低 ⇒ 草稿器没毛病, 天花板是②(质量线), 抬①只能让草稿器改盯底座;\n");
    printf("      ③也低      ⇒ 病在草稿器实现或它吃的量化隐态, 要金标夹具再分一刀。\n");
    printf("      ①与 ②×③ 差很多 ⇒ 两条偏离不独立(有共同成因), 也值得回去看。\n");
    free(row); free(mtok); free(dtok); fclose(fa);
    return 0;
}
