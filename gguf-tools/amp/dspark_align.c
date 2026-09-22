/* dspark_align.c — DSpark 草稿器对齐部署底座的闭式解(mtp.md M6, 2026-09-16)。
 *
 * 为什么要它: 草稿器是照**原始 FP 模型**训的, 我们部署的是量化+反修的底座。实测"草稿器首位 ↔
 * 底座 argmax"一致率只有 0.52(512 位置判决料), 而底座对 FP 原模型的 Same top 是 0.72 ——
 * 0.72 × 0.85(草稿器自身命中) ≈ 0.61, 与在线量到的 p1 对得上。也就是说**草稿器没毛病, 是底座漂移
 * 把它的收益吃掉了**, 而 0.72 是它的天花板。接受率上不去, 投机就永远赢不了纯解码(mtp.md §5 的字节账)。
 *
 * 这一片做的事: 让草稿器改盯**底座**说话。关键观察是两边的 logits 由**同一个出口头**算
 * (官方 forward_head 借主模型的 head), 所以只要把草稿器喂给头的隐态掰到主模型喂给头的那个,
 * logits 自然对齐 —— 于是问题塌缩成一个最普通的最小二乘:
 *
 *     min ‖ X·(I + Bᵀ A) − Y ‖²     X = 草稿器出口隐态, Y = 主模型出口隐态(同一位置)
 *
 * 形式与反修放大器一模一样(引擎 ds4_gpu_v41_amp_apply_tensor: T = x·Bᵀ, y += T·A), 所以
 * **产物是几十 MB 的边车, 主模型一个字节不碰, 不重转 GGUF**。
 *
 * 怎么解(全闭式, 无 SGD/无 epoch):
 *   ★关键是 n ≪ D★(取料 512~8192 位置, D = 5120) —— 所以走**行空间**而不是列空间:
 *     M = Xᵀ(X Xᵀ + λI)⁻¹ R      (R = Y − X)
 *   这与 (XᵀX + λI)⁻¹XᵀR 是同一个解(推导: 两边同乘), 但解的是 **n×n** 的方程而不是 5120×5120,
 *   而且 M 天然秩 ≤ n。整份 M(5120×5120 = 105 MB)从头到尾**不materialize**。
 *   秩截断取 RRR 的最优形态: 对拟合值 Ŷ = X·M 做 SVD, 留前 K 个右奇异方向 ——
 *   Ŷ 只有 n 行, 所以走 ŶŶᵀ(n×n)的特征分解, 不做 D×D。
 *
 * ★λ 与 K 靠留出行选★: 取料 n 行按序切成拟合/验证两段(默认后 20% 作验证), 扫 λ×K 取验证残差最小的。
 * 512 行拟合 5120 维是**欠定**的, 岭必须压得住 —— 不留验证行就一定过拟合, 而过拟合不会报错,
 * 只会让引擎侧的 p1 一点不涨甚至更差。
 *
 * ★判据在引擎侧, 不在这里★: 这里只报残差下降; 真判决是引擎挂上边车后重测
 * "草稿器首位 ↔ 底座 argmax"一致率(--dspark-capture 那一行)。残差降了而 p1 不涨是可能的
 * (隐态对齐 ≠ argmax 对齐), 所以别拿残差当结论。
 *
 * 用法: dspark_align <pairs.bin> <out.bin> [--hdiag <pairs.bin.hdiag>] [--k-list 64,128,256] [--lam-list 0.1,1,10] [--val 0.2]
 * ★--hdiag 必给★: 不给就是纯 L2 度量, 09-16 判过负(留出一致率不升反降), 只作对照。
 * 编译: make -C gguf-tools dspark_align */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "linalg_small.h"

/* pos0 = 第 0 对对应的主模型位置(引擎 09-18 起写 1); 这里不用它 —— 解算只看 (X, Y) 对, 不看位置。
 * ★pos0 == 0 的老料是错位的(core_v41_dcap.c 文件头), 拿它解出来的边车就是 09-16/09-17 判负那两份。★ */
typedef struct { char magic[4]; uint32_t d, n, pos0; } dcap_hdr;

static double *xmal(size_t n) {
    double *p = (double *)malloc(n * sizeof(double));
    if (!p) { fprintf(stderr, "内存不够: %zu 个 double\n", n); exit(1); }
    return p;
}

/* C = A·Bᵀ, A 是 (ra×k), B 是 (rb×k), 结果 (ra×rb) 行主序 */
static void mm_abt(const double *A, const double *B, double *C, int ra, int rb, int k) {
    for (int i = 0; i < ra; i++)
        for (int j = 0; j < rb; j++) {
            const double *a = A + (size_t)i * k, *b = B + (size_t)j * k;
            double s = 0;
            for (int t = 0; t < k; t++) s += a[t] * b[t];
            C[(size_t)i * rb + j] = s;
        }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: dspark_align <pairs.bin> <out.bin> [--k-list a,b] [--lam-list a,b] [--val f]\n"); return 1; }
    const char *inp = argv[1], *outp = argv[2], *hdiag_path = NULL;
    /* ★--rows: 只用前 N 行取料★ 这个解算器走行空间(n×n), 代价是 O(n²·D) 的 Gram + O(n²·K) 的特征分解 ——
     * n=512 时几秒, n=8191 时 2.2e11 次乘加, 纯 C 单线程要几十分钟, 而且 n×n 的几块 double 就 1.4 GB。
     * 实撞: 直接喂 8191 行, 进程没出一行日志就没了。默认 2048 是"够拟合 5120 维又跑得动"的折中。 */
    int max_rows = 2048;
    int klist[8] = { 64, 128, 256, 0 }, nk = 3;
    double lamlist[8] = { 0.1, 1.0, 10.0, 0 }; int nl = 3;
    double valfrac = 0.2;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--k-list") && i + 1 < argc) {
            nk = 0; char *s = strdup(argv[++i]), *t = strtok(s, ",");
            while (t && nk < 8) { klist[nk++] = atoi(t); t = strtok(NULL, ","); }
            free(s);
        } else if (!strcmp(argv[i], "--lam-list") && i + 1 < argc) {
            nl = 0; char *s = strdup(argv[++i]), *t = strtok(s, ",");
            while (t && nl < 8) { lamlist[nl++] = atof(t); t = strtok(NULL, ","); }
            free(s);
        } else if (!strcmp(argv[i], "--val") && i + 1 < argc) valfrac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--hdiag") && i + 1 < argc) hdiag_path = argv[++i];
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) max_rows = atoi(argv[++i]);
    }

    FILE *f = fopen(inp, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", inp); return 1; }
    dcap_hdr h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, "DCAP", 4)) { fprintf(stderr, "%s 不是取料文件\n", inp); return 1; }
    const int D = (int)h.d, n = (int)h.n;
    if (D <= 0 || n <= 2) { fprintf(stderr, "取料太小(D=%d n=%d)\n", D, n); return 1; }
    if (h.pos0 == 0) {   /* 禁兜底: 错位料解出来的边车挂上去只会把草稿变差(09-16/09-17 各撞一次), 直接停 */
        fprintf(stderr, "★%s 是 09-18 修尺之前的取料(pos0=0, 草稿首位错位), 不许拿它解边车 —— 用修过的引擎重取★\n", inp);
        return 1;
    }
    printf("[align] 取料 %s: %d 位置 × %d 维\n", inp, n, D);

    double *X = xmal((size_t)n * D), *R = xmal((size_t)n * D);
    float *buf = (float *)malloc((size_t)D * 2 * sizeof(float));
    /* ★进门先体检★(2026-09-16 实撞): 草稿器偶发整块非有限(mtp.md §3.3 记过 "草稿首位 = 0, conf = nan"),
     * 一行 NaN 会顺着 Gram → Cholesky → 特征分解把**整个解**变成 NaN, 而且中间一步都不报错 ——
     * 上一版就是这么打出九行 "验证残差 nan" 的。坏行直接丢掉并报数量: 丢几行不影响闭式解,
     * 但**不知道丢了多少**就会把"取料坏了"误判成"方法不行"。 */
    int nok = 0, nbad = 0;
    for (int i = 0; i < n; i++) {
        if (fread(buf, sizeof(float), (size_t)D * 2, f) != (size_t)D * 2) { fprintf(stderr, "取料在第 %d 位截断\n", i); return 1; }
        int good = 1;
        for (int d = 0; d < 2 * D; d++) if (!isfinite(buf[d])) { good = 0; break; }
        if (!good) { nbad++; continue; }
        for (int d = 0; d < D; d++) {
            X[(size_t)nok * D + d] = buf[d];
            R[(size_t)nok * D + d] = (double)buf[D + d] - (double)buf[d];   /* R = Y − X: 要拟合的修正量 */
        }
        nok++;
    }
    free(buf); fclose(f);
    if (nbad) printf("[align] ★丢掉 %d 行非有限取料(留 %d 行)★ —— 草稿器那几轮是坏的, 值得回去查\n", nbad, nok);
    if (max_rows > 0 && nok > max_rows) {
        printf("[align] 取料 %d 行, 只用前 %d 行(--rows 改; 行空间解是 O(n²·D), 见 max_rows 注释)\n", nok, max_rows);
        nok = max_rows;
    }
    if (nok <= 2) { fprintf(stderr, "可用取料太少(%d 行)\n", nok); return 1; }

    /* ★出口度量白化(mtp-1.md M6′)★
     * 09-16 那次判负: 纯 L2 解出来的边车挂上去, 留出首位一致率 0.4570 → 0.4336(β 越小越接近不挂,
     * 但从来没超过) —— 不是幅度过头, 是**目标函数错了**。草稿器与主模型的 logits 由同一个出口头 W 算,
     * 所以该量的是 ‖W·(h_d−h_m)‖ 而不是 ‖h_d−h_m‖: 头基本不看的那些维度, 对齐得再准也不改 argmax;
     * 而纯 L2 恰恰会把力气花在方差大的方向上。
     * 这里用 G = WᵀW 的**对角线**(引擎 --dspark-capture 顺带落的 <pairs>.hdiag, 每列的平方和)作一阶近似:
     * 把 R 的第 d 列乘 w_d = sqrt(cs_d)(归一化到均值 1, 免得 λ 的量纲跟着变), 照旧解, 最后把 A 的第 d 列
     * 除回去 —— 等价于在 diag(G) 这个度量里做同一套岭 + 秩截断。
     * ★不给 --hdiag 就退回纯 L2★, 并大声说一句: 那条路 09-16 判过负, 只作对照用。 */
    double *wcol = xmal((size_t)D);
    for (int d = 0; d < D; d++) wcol[d] = 1.0;
    if (hdiag_path) {
        FILE *fh = fopen(hdiag_path, "rb");
        float *cs = (float *)malloc((size_t)D * sizeof(float));
        if (fh && cs && fread(cs, sizeof(float), (size_t)D, fh) == (size_t)D) {
            double mean = 0;
            for (int d = 0; d < D; d++) { wcol[d] = sqrt((double)(cs[d] > 0 ? cs[d] : 0)); mean += wcol[d]; }
            mean /= D;
            if (mean > 0) for (int d = 0; d < D; d++) wcol[d] /= mean;
            double lo = 1e30, hi = 0;
            for (int d = 0; d < D; d++) { if (wcol[d] < lo) lo = wcol[d]; if (wcol[d] > hi) hi = wcol[d]; }
            printf("[align] 出口度量已挂: 列权重 min %.3f / max %.3f(均值 1)\n", lo, hi);
            for (int i = 0; i < nok; i++)
                for (int d = 0; d < D; d++) R[(size_t)i * D + d] *= wcol[d];
        } else printf("[align] ★--hdiag 读不到 %s, 退回纯 L2(那条判过负)★\n", hdiag_path);
        if (fh) fclose(fh);
        free(cs);
    } else printf("[align] ★没给 --hdiag: 走纯 L2 度量 —— 09-16 判过负, 只作对照★\n");

    const int nrows = nok;   /* 体检后真正可用的行数(下面一律用它, 别再碰 n —— n 是文件里写的总数) */
    const int nval = (int)((double)nrows * valfrac), nfit = nrows - nval;
    if (nfit < 16) { fprintf(stderr, "拟合行太少(%d)\n", nfit); return 1; }
    printf("[align] 拟合 %d 行 / 验证 %d 行(按序切, 后段作验证)\n", nfit, nval);

    /* 基线: 不做任何修正时验证段的相对残差 = ‖R_val‖² / ‖Y_val‖²; 这里用 ‖R‖² 当分母的参照量 */
    double base_val = 0;
    for (int i = nfit; i < nrows; i++)
        for (int d = 0; d < D; d++) base_val += R[(size_t)i * D + d] * R[(size_t)i * D + d];

    double *K0 = xmal((size_t)nfit * nfit);          /* X_fit X_fitᵀ */
    mm_abt(X, X, K0, nfit, nfit, D);
    double trd = 0; for (int i = 0; i < nfit; i++) trd += K0[(size_t)i * nfit + i];
    const double scale = trd / nfit;

    double *Gram = xmal((size_t)nfit * nfit), *G = xmal((size_t)nfit * D);
    double *Yh = xmal((size_t)nfit * D), *C = xmal((size_t)nfit * nfit);
    double *U = xmal((size_t)nfit * nfit), *ev = xmal((size_t)nfit);
    double *V = xmal((size_t)D * 8), *GV = xmal((size_t)nfit * 8);   /* 按最大 K 重分配, 见下 */
    int kmax = 0; for (int i = 0; i < nk; i++) if (klist[i] > kmax) kmax = klist[i];
    if (kmax > nfit) kmax = nfit;
    free(V); free(GV);
    V = xmal((size_t)D * kmax); GV = xmal((size_t)nfit * kmax);
    double *Bbest = xmal((size_t)kmax * D), *Abest = xmal((size_t)kmax * D);
    int kbest = 0; double lambest = 0, valbest = 1e300;

    for (int li = 0; li < nl; li++) {
        memcpy(Gram, K0, (size_t)nfit * nfit * sizeof(double));
        const double lam = lamlist[li] * scale;
        for (int i = 0; i < nfit; i++) Gram[(size_t)i * nfit + i] += lam;
        memcpy(G, R, (size_t)nfit * D * sizeof(double));
        if (chol_solve_spd(Gram, nfit, G, D)) { printf("[align] λ=%g 不正定, 跳过\n", lamlist[li]); continue; }
        /* Ŷ = X M = (X Xᵀ) G */
        for (int i = 0; i < nfit; i++)
            for (int d = 0; d < D; d++) {
                double s = 0;
                for (int j = 0; j < nfit; j++) s += K0[(size_t)i * nfit + j] * G[(size_t)j * D + d];
                Yh[(size_t)i * D + d] = s;
            }
        mm_abt(Yh, Yh, C, nfit, nfit, D);
        sym_eig_topk(C, nfit, kmax, U, ev, 0, 20260916ull);
        for (int ki = 0; ki < nk; ki++) {
            int K = klist[ki]; if (K > nfit) K = nfit;
            /* V[:,j] = Ŷᵀ u_j / ‖·‖ */
            for (int j = 0; j < K; j++) {
                double nrm = 0;
                for (int d = 0; d < D; d++) {
                    double s = 0;
                    for (int i = 0; i < nfit; i++) s += Yh[(size_t)i * D + d] * U[(size_t)j * nfit + i];
                    V[(size_t)d * K + j] = s; nrm += s * s;
                }
                nrm = sqrt(nrm) + 1e-30;
                for (int d = 0; d < D; d++) V[(size_t)d * K + j] /= nrm;
            }
            /* GV = G·V (nfit×K); B_row = Xᵀ·GV (D×K) ⇒ B[K][D] = B_rowᵀ; A[K][D] = Vᵀ */
            for (int i = 0; i < nfit; i++)
                for (int j = 0; j < K; j++) {
                    double s = 0;
                    for (int d = 0; d < D; d++) s += G[(size_t)i * D + d] * V[(size_t)d * K + j];
                    GV[(size_t)i * K + j] = s;
                }
            double *Bk = xmal((size_t)K * D), *Ak = xmal((size_t)K * D);
            for (int j = 0; j < K; j++)
                for (int d = 0; d < D; d++) {
                    double s = 0;
                    for (int i = 0; i < nfit; i++) s += X[(size_t)i * D + d] * GV[(size_t)i * K + j];
                    Bk[(size_t)j * D + d] = s;
                    Ak[(size_t)j * D + d] = V[(size_t)d * K + j];
                }
            /* 验证段残差: ‖X_val·Bᵀ·A − R_val‖² */
            double vres = 0;
            double *t = xmal((size_t)K);
            for (int i = nfit; i < nrows; i++) {
                const double *xi = X + (size_t)i * D, *ri = R + (size_t)i * D;
                for (int j = 0; j < K; j++) {
                    double s = 0; const double *b = Bk + (size_t)j * D;
                    for (int d = 0; d < D; d++) s += xi[d] * b[d];
                    t[j] = s;
                }
                for (int d = 0; d < D; d++) {
                    double s = 0;
                    for (int j = 0; j < K; j++) s += t[j] * Ak[(size_t)j * D + d];
                    const double e = s - ri[d]; vres += e * e;
                }
            }
            free(t);
            printf("[align] λ=%-8g K=%-5d 验证残差 %.6f(不修正 = 1.0)\n", lamlist[li], K, vres / (base_val + 1e-30));
            fflush(stdout);
            if (vres < valbest) { valbest = vres; kbest = K; lambest = lamlist[li];
                                  memcpy(Bbest, Bk, (size_t)K * D * sizeof(double));
                                  memcpy(Abest, Ak, (size_t)K * D * sizeof(double)); }
            free(Bk); free(Ak);
        }
    }
    if (!kbest) { fprintf(stderr, "没有可用的解(全部不正定?)\n"); return 1; }
    printf("[align] ★选中 λ=%g K=%d, 验证残差 %.6f(不修正 = 1.0)★\n", lambest, kbest, valbest / (base_val + 1e-30));
    if (valbest / (base_val + 1e-30) > 0.98)
        printf("[align] ★残差几乎没降 —— 取料太少或 λ 全压死了, 别急着上引擎★\n");

    /* 把 A 的列除回去: 解是在白化空间里做的, 引擎那边吃的是原空间的 x(见上面 wcol 那段) */
    for (int j = 0; j < kbest; j++)
        for (int d = 0; d < D; d++) if (wcol[d] > 0) Abest[(size_t)j * D + d] /= wcol[d];

    FILE *fo = fopen(outp, "wb");
    if (!fo) { fprintf(stderr, "写不了 %s\n", outp); return 1; }
    struct { char magic[4]; uint32_t d, k, rsv; } oh = { { 'D','S','P','A' }, (uint32_t)D, (uint32_t)kbest, 0 };
    float *row = (float *)malloc((size_t)D * sizeof(float));
    int ok = fwrite(&oh, sizeof oh, 1, fo) == 1;
    for (int j = 0; j < kbest && ok; j++) {   /* A[K][D] 在前, B[K][D] 在后(引擎按这个序读) */
        for (int d = 0; d < D; d++) row[d] = (float)Abest[(size_t)j * D + d];
        ok = fwrite(row, sizeof(float), (size_t)D, fo) == (size_t)D;
    }
    for (int j = 0; j < kbest && ok; j++) {
        for (int d = 0; d < D; d++) row[d] = (float)Bbest[(size_t)j * D + d];
        ok = fwrite(row, sizeof(float), (size_t)D, fo) == (size_t)D;
    }
    fclose(fo); free(row);
    if (!ok) { fprintf(stderr, "落盘失败\n"); return 1; }
    printf("[align] 落盘 %s: D=%d K=%d(A 在前 B 在后, 各 K×D f32)\n", outp, D, kbest);
    printf("[align] ★判决不在这里★: 拿它上引擎 --draft-amp 重跑 --dspark-capture, 看首位一致率涨没涨。\n");
    return 0;
}
