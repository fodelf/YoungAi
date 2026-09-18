/* dspark_agree.c — 把"投机接受率上限"的三个一致率一次量清楚(mtp-1.md M0′(c), 2026-09-17)。
 *
 * 为什么必须是三个数而不是一个: 草稿塔(DSpark 三塔)是照**原始 FP 模型**训的, 出厂 FP4 直接透传;
 * 而我们部署的主干是 VQ 1.5 bpw + 反修。于是引擎在线量到的"接受率"其实是两条偏离叠在一起:
 *     p1 ≈ (草稿器↔FP 原模型) × (底座↔FP 原模型)
 * 只量到一个 0.56 根本分不清该怪谁 —— 09-16 就是拿这一个数当依据去"把草稿器掰向底座", 判负两次。
 * 三个数一起摆出来才有下一步:
 *   ①草稿器↔底座 = 在线 p1 的上限(引擎 --dspark-capture 直接报)
 *   ②底座↔FP 锚  = 量化+反修还了多少原模型(= 五指标里的 Same top, 质量线的数)
 *   ③草稿器↔FP 锚 = **草稿器自己准不准**(它本来就该对着 FP 说话)
 * 读法:
 *   ③高(≈0.85)而①低 ⇒ 草稿器没毛病, 天花板就是②, 要抬接受率只能抬②(质量线)或重训草稿器;
 *   ③也低(≈0.6)     ⇒ 病在**我们这边的草稿器实现**(取 main_hidden 的位置/类型分发/块语义),
 *                      那是工程 bug, 修了①直接跟着涨 —— 这条路比动量化便宜得多。
 *
 * 输入: ①引擎 --dspark-capture 落的 pairs.bin(尾部存着每位的"底座 argmax"与"草稿首位"两串 token)
 *       ②FP 锚(DQA2 格式, 与 caliper 判决尺同一份)
 * 用法: dspark_agree <pairs.bin> <anchor_j_s8192.bin>
 * 编译: make -C gguf-tools dspark_agree
 *
 * ★口径对齐是这把尺的命门★: 取料第 i 位喂 ids[i]、预测 ids[i+1], 锚的第 i 行也是"位置 i 的 logits";
 * 两边错一行就会得出一个看着合理其实全假的数。所以下面按 min(n_取料, S_锚) 逐行对, 并把两个 n 都打出来。 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct { char magic[4]; uint32_t d, n, rsv; } dcap_hdr;

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: dspark_agree <pairs.bin> <anchor.bin>\n"); return 1; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "%s 打不开\n", argv[1]); return 1; }
    dcap_hdr h;
    if (fread(&h, sizeof h, 1, fp) != 1 || memcmp(h.magic, "DCAP", 4)) { fprintf(stderr, "%s 不是取料文件\n", argv[1]); return 1; }
    const long long D = h.d, n = h.n;
    /* 两串 token 在文件尾: [主模型 argmax i32 × n][草稿首位 i32 × n] */
    if (fseek(fp, (long)(sizeof h + (long long)n * D * 2 * 4), SEEK_SET) != 0) { fprintf(stderr, "定位失败\n"); return 1; }
    int32_t *mtok = malloc((size_t)n * 4), *dtok = malloc((size_t)n * 4);
    if (!mtok || !dtok || fread(mtok, 4, (size_t)n, fp) != (size_t)n || fread(dtok, 4, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "token 串读不全(取料被截断?)\n"); return 1;
    }
    fclose(fp);

    /* ★顺手量一眼两边出口隐态的几何关系★(2026-09-17): 取料文件里每位都存着
     * X = 草稿器喂给出口头的隐态、Y = 主模型喂给同一个头的隐态。它们过的是**同一个头**,
     * 所以 argmax 差多少, 根子上就是这两个向量差多少。
     *   cos 高、范数比 ≈1 ⇒ 草稿器方向是对的, 差的是细节(它只是个小模型);
     *   cos 低或范数比离 1 很远 ⇒ 出口那一段有结构性问题(norm/缩放/取错了哪一层),
     *     那不是"模型弱", 是实现欠账 —— 而且这种情况下拿线性映射去对齐当然解不出来。 */
    {
        FILE *fx = fopen(argv[1], "rb");
        float *xb = (float *)malloc((size_t)D * 2 * 4);
        if (fx && xb && fseek(fx, sizeof h, SEEK_SET) == 0) {
            double csum = 0, rsum = 0; long long cn = 0;
            const long long probe = n < 512 ? n : 512;   /* 512 位够看形态, 不用读整份 */
            for (long long i = 0; i < probe; i++) {
                if (fread(xb, 4, (size_t)D * 2, fx) != (size_t)D * 2) break;
                double dot = 0, nx = 0, ny = 0;
                for (long long d = 0; d < D; d++) { dot += (double)xb[d] * xb[D + d]; nx += (double)xb[d] * xb[d]; ny += (double)xb[D + d] * xb[D + d]; }
                if (nx > 0 && ny > 0) { csum += dot / (sqrt(nx) * sqrt(ny)); rsum += sqrt(nx) / sqrt(ny); cn++; }
            }
            if (cn) printf("[agree] 出口隐态(前 %lld 位): cos(草稿, 主模型) = %.4f, 范数比 ‖草稿‖/‖主模型‖ = %.4f\n",
                           cn, csum / cn, rsum / cn);
        }
        if (fx) fclose(fx);
        free(xb);
    }

    FILE *fa = fopen(argv[2], "rb");
    if (!fa) { fprintf(stderr, "%s 打不开\n", argv[2]); return 1; }
    uint32_t hd[8];
    if (fread(hd, 4, 8, fa) != 8 || hd[0] != 0x32415144u) { fprintf(stderr, "%s 不是 DQA2 锚\n", argv[2]); return 1; }
    const long long S = hd[1], HCM = hd[2], DIM = hd[3], NL = hd[4], VOCAB = hd[5], NACT = hd[6];
    fseek(fa, 8, SEEK_CUR);   /* idh */
    const long long skip = NL * S * DIM * 4 + NL * S * NACT * 4 * 2 + NL * S * HCM * DIM * 4;
    if (fseek(fa, (long)skip, SEEK_CUR) != 0) { fprintf(stderr, "锚定位失败\n"); return 1; }

    const long long rows = n < S ? n : S;
    printf("[agree] 取料 %lld 位 / 锚 %lld 位(词表 %lld) ⇒ 逐行对前 %lld 位\n", n, S, VOCAB, rows);
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
        if (fread(row, 4, (size_t)VOCAB, fa) != (size_t)VOCAB) { fprintf(stderr, "锚第 %lld 行截断\n", i); return 1; }
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
    printf("  ②部署底座 ↔ FP 锚  = %.4f   (= 量化+反修还了多少原模型)\n", (double)af / (double)rows);
    printf("  ③草稿器 ↔ FP 锚    = %.4f   (= 草稿器自己准不准; 它本来就对着 FP 说话)\n", (double)df / (double)rows);
    printf("\n读法: ③≈0.85 而①低 ⇒ 草稿器没毛病, 天花板是②(质量线);\n");
    printf("      ③也低      ⇒ 病在我们这边的草稿器实现, 那是工程 bug, 修了①直接跟着涨。\n");
    printf("      ①与 ②×③ 差很多 ⇒ 两条偏离不独立(有共同成因), 也值得回去看。\n");
    free(row); free(mtok); free(dtok); fclose(fa);
    return 0;
}
