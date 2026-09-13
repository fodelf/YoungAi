/* finetune_solve.c — 微调侧车解算器(2026-09-08 用户定案的第三个文件)。
 *
 * 干什么: 同一个部署模型(量化 GGUF + zchain)跑两遍 —— 一遍让它读到复盘事后上下文
 * (教师), 一遍不读(学生) —— 逐层把两者 routed 输出的差, 用一个低秩映射从学生自己的
 * 输入 x 里预测出来。产物是独立的 DQZ2 文件, 引擎 --finetune 挂上, 与 zchain 一起生效。
 *
 *     R = routed_教师 − routed_学生 ≈ U·diag(z)·Vᵀ·x_学生
 *
 * 为什么不用 FP 教师、不用建锚: zchain 已经把"还原原始模型"这件事做完并冻结了; 微调
 * 要学的是"看过复盘之后该怎么判", 那个目标就在同一个部署模型自己身上, 让它读一遍上下文
 * 就有了。于是全链只剩两遍引擎前向(--cap-dir, 7.5k token 各一分钟)+ 本解算器, 不再有
 * 30 GB 的锚和两小时的 43 层重解。
 *
 * 取料 = 引擎 --cap-dir 的原样输出(部署同路, 铁律"捕获必须与部署同路"):
 *   raw_ffn_in_L{L}   f16 [n×4096]  x̂ = MoE 输入(post ffn_norm)
 *   raw_ffn_out_L{L}  f16 [n×4096]  routed 输出 ★zchain 之后★(捕获点见 core_gpu_prefill_ffn.c)
 * 教师序列比学生多一段只给教师看的上下文, 行号对应关系由 <ids>.layout 的
 *   xshift <S0> <N>
 * 给出: 锚行 <S0 → 学生同号; [S0,S0+N) 学生没有; ≥S0+N → 学生行 −N(与 row_layout.inc.c 同一映射)。
 *
 * ★注入点(2026-09-08 首跑定罪后改)★ --mode:
 *   rte(默认) 打【路由 logits】, 产 type8: δlogits[e] = Σ_c U[e][c]·z[c]·tanh((V[:,c]·x)/s)
 *   routed    打【routed 输出】, 产 type6(首跑实测全负, 保留只为复现那次判决)
 * 为什么换: 首跑实测 —— 读了上下文后 x 只动一点(逐行 cos 0.91)、路由 logits 几乎没动
 * (cos 0.94~0.999), 但每层有 1.3~1.7 个专家被换掉, 换进换出的专家输出毫不相干 ⇒ routed 差到
 * cos 0.33~0.54。一个作用在 x 上的平滑低秩加性映射, 原理上表示不了"专家被换掉"这种离散跳变,
 * 所以 routed 模式 43 层 held-out 全负(λ 调到不炸也只剩 +0.8%)。改打路由 = 直接改"选谁", 绕开跳变。
 *
 * 用法: finetune_solve --teacher DIR --student DIR --layout FILE --out FILE
 *                      [--mode rte|routed] [--rank 16] [--lambda 0.05] [--layers a-b] [--min-gain 0.02]
 * 判据(rte): held-out 行上【前 6 专家与教师的重合率】提升多少 —— 这才是"专家被换掉"这件事本身,
 *   logits 残差降多少只是过程量。routed 模式仍用残差能量下降比。不到 --min-gain 的层不注入;
 *   全 43 层都没过 = 不产文件(退出码非 0), 今晚不上线。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdarg.h>
#include "../../ds4_z.h"
#include "../../src/common/ds4_float.h"
#include "../../src/common/ds4_amp_fmt.h"
#include "../../src/common/ds4_gguf.h"
#include "../../src/common/ds4_quantfmt.h"

#define DIM 4096u
#define NEXP 256u
#define TOPK 6u

static void die(const char *fmt, ...);
#include "finetune_sft_target.inc.c"   /* --mode sft 的靶: 损失对 routed 的梯度 */

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "★"); vfprintf(stderr, fmt, ap); fprintf(stderr, "★\n");
    va_end(ap); exit(1);
}

/* f16 分片读成 f32。返回行数; buf 由调用方 free。 */
static float *read_f16_rows(const char *dir, const char *name, int L, long *rows_out, uint32_t w) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s_L%d", dir, name, L);
    FILE *f = fopen(p, "rb");
    if (!f) die("捕获打不开: %s", p);
    fseeko(f, 0, SEEK_END);
    long long sz = ftello(f);
    fseeko(f, 0, SEEK_SET);
    long rows = (long)(sz / 2 / w);
    if (rows <= 0) die("%s 行数 0(文件 %lld 字节)", p, sz);
    uint16_t *h = malloc((size_t)rows * w * 2);
    float *o = malloc((size_t)rows * w * sizeof(float));
    if (!h || !o) die("内存不足(%s, %ld 行)", p, rows);
    if (fread(h, 2, (size_t)rows * w, f) != (size_t)rows * w) die("%s 读不满", p);
    fclose(f);
    for (size_t i = 0; i < (size_t)rows * w; i++) o[i] = ds4_f16_to_f32(h[i]);
    free(h);
    *rows_out = rows;
    return o;
}

/* 一行 logits 的前 TOPK 专家, 与另一行的前 TOPK 有几个重合。
 * ★这是近似★: 真实路由还有分组/偏置(ffn_exp_probs_b), 这里只按裸 logits 取前 6。
 * 但教师侧与学生侧用同一把尺, 所以"重合率提升了多少"这个相对量仍然可信。 */
static void topk_ids(const float *v, uint32_t n, uint32_t k, int *out) {
    for (uint32_t a = 0; a < k; a++) {
        int best = -1;
        for (uint32_t e = 0; e < n; e++) {
            int used = 0;
            for (uint32_t b = 0; b < a; b++) if (out[b] == (int)e) { used = 1; break; }
            if (!used && (best < 0 || v[e] > v[best])) best = (int)e;
        }
        out[a] = best;
    }
}
static double topk_overlap(const float *a, const float *b) {
    int ia[TOPK], ib[TOPK];
    topk_ids(a, NEXP, TOPK, ia); topk_ids(b, NEXP, TOPK, ib);
    int c = 0;
    for (uint32_t i = 0; i < TOPK; i++) for (uint32_t j = 0; j < TOPK; j++) if (ia[i] == ib[j]) { c++; break; }
    return (double)c;
}

/* 从 <ids>.layout 读 xshift 与正文块。缺任一项都硬停 —— 猜行域是 2026-08-29 的老坑。 */
static void read_layout(const char *path, int *s0, int *nctx, int *body_lo, int *body_n) {
    FILE *f = fopen(path, "r");
    if (!f) die("布局打不开: %s", path);
    *s0 = *nctx = *body_lo = *body_n = -1;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        long a, b;
        char nm[128];
        if (line[0] == '#') continue;
        if (sscanf(line, "xshift %ld %ld", &a, &b) == 2) { *s0 = (int)a; *nctx = (int)b; continue; }
        if (sscanf(line, "%127s %ld %ld", nm, &a, &b) == 3 && !strcmp(nm, "hindsight")) {
            *body_lo = (int)a; *body_n = (int)b;
        }
    }
    fclose(f);
    if (*s0 < 0 || *nctx <= 0) die("%s 里没有 xshift 行(教师/学生行偏移)", path);
    if (*body_lo < 0 || *body_n <= 0) die("%s 里没有 hindsight 块(正文行域)", path);
}

int main(int argc, char **argv) {
    const char *td = NULL, *sd = NULL, *lay = NULL, *outp = NULL, *lrange = NULL, *mode = "rte";
    const char *tkp = NULL, *ggp = NULL;   /* --mode sft: topk 产物 + 取输出头权重的 GGUF */
    float eta = 1.0f;                      /* 步长: 与 α/rms 一起吸收(见 inc.c 近似 2、3) */
    int sft_layer = 42;                    /* --layer: 注入哪一层(见下面残差直连那段) */
    const char *sft_rows = NULL;           /* --rows a:b,c:d: 只把这些行算进靶(报告段) */
    uint32_t rank = 16;
    float lambda = 0.05f, min_gain = 0.02f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--teacher") && i + 1 < argc) td = argv[++i];
        else if (!strcmp(argv[i], "--student") && i + 1 < argc) sd = argv[++i];
        else if (!strcmp(argv[i], "--layout") && i + 1 < argc) lay = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) lambda = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--min-gain") && i + 1 < argc) min_gain = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) lrange = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) tkp = argv[++i];
        else if (!strcmp(argv[i], "--gguf") && i + 1 < argc) ggp = argv[++i];
        else if (!strcmp(argv[i], "--eta") && i + 1 < argc) eta = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) sft_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) sft_rows = argv[++i];
        else die("未知参数: %s", argv[i]);
    }
    const int sft = !strcmp(mode, "sft");
    const int rte = !strcmp(mode, "rte");
    if (!rte && !sft && strcmp(mode, "routed")) die("--mode 只认 rte / routed / sft");
    if (!sd || !outp || (!sft && (!td || !lay)) || (sft && (!tkp || !ggp)))
        die("用法: finetune_solve --student DIR --out FILE\n"
            "  教师差靶: --teacher DIR --layout FILE [--mode rte|routed]\n"
            "  SFT 梯度靶: --mode sft --topk FILE --gguf FILE [--eta 1.0] [--layer 42]\n"
            "  公共: [--rank 16] [--lambda 0.05] [--layers a-b] [--min-gain 0.02]");
    if (rank == 0 || rank > DS4_AMP_ZK_MAX) die("--rank %u 超范围(1..%d)", rank, (int)DS4_AMP_ZK_MAX);

    /* ---- --mode sft: 一步低秩 SFT。--layer 选注入层(默认 42) ----
     * L42 是严格闭式。★往前几层用残差直连近似★(2026-09-10, L42 单点到顶之后):
     *   hc_next[c] = after_attn_hc[c] + split[c]·(routed+shared)
     * ⇒ routed_L 经残差直连进入后面每一层的 hc, 而残差是深度网络里梯度流的主干道。
     * 取直连路径: ∂L/∂routed_L ≈ (逐 token 正标量) · ∂L/∂routed_42 —— ★方向与 L42 相同★,
     * 差的那个标量按 inc.c 的近似 3 本来就吸收进 η 了。所以靶 R 一个字节都不用重算,
     * 换的只是 X(该层的 x)。★新容量正来自这里★: 各层梯度方向近似相同, 但各层的 x
     * 携带的信息不同, 低秩映射能表达的函数就不同 —— 这是单层容量到顶后唯一不用写
     * 反向传播就能拿到的杠杆。
     * ★这个近似糙在哪, 说清楚★: 丢掉了穿过 attention 与后续 FFN 的非线性路径, 层数
     * 越往前丢得越多。所以它不是"理论上更对", 而是"便宜且可证伪" —— 解出来挂上去
     * 看判据降不降, 降不动就说明该层的直连近似不成立, 不要硬解。 */
    if (sft) {
        sft_topk tk; sft_topk_read(tkp, &tk);
        ds4_gguf gg; char gerr[256];
        if (ds4_gguf_open(&gg, ggp, gerr, sizeof gerr)) die("--gguf %s", gerr);
        const ds4_gguf_tensor *wt = ds4_gguf_find(&gg, "output.weight");
        const ds4_gguf_tensor *nt = ds4_gguf_find(&gg, "output_norm.weight");
        if (!wt || !nt) die("%s 里没有 output.weight / output_norm.weight", ggp);
        if (wt->type != 8u) die("output.weight 类型 %u 不是 Q8_0 — 这条靶目前只解 Q8_0 输出头", wt->type);
        if (wt->ne[0] != DIM) die("output.weight ne[0]=%llu 不是 %u", (unsigned long long)wt->ne[0], DIM);
        const uint8_t *wq8 = gg.map + gg.data0 + wt->off;
        const float *onorm = (const float *)(const void *)(gg.map + gg.data0 + nt->off);

        long xrows = 0;
        if (sft_layer < 0 || sft_layer > 42) die("--layer %d 超范围(0..42)", sft_layer);
        float *Xall = read_f16_rows(sd, "raw_ffn_in", sft_layer, &xrows, DIM);
        float *X = malloc((size_t)tk.n * DIM * sizeof(float));
        float *R = malloc((size_t)tk.n * DIM * sizeof(float));
        if (!X || !R) die("sft 取料内存不足");
        /* --eval-topk 的行号与捕获行号同一口径(都来自同一条 ids), 所以 xshift=0;
         * 留这个参数是为了将来两边喂不同 ids 时能对齐, 不是给它默认值糊过去。 */
        /* --rows: 解析成逐行的 keep 掩码。格式与 anchor_metrics --rows 同源(a:b,c:d,...),
         * 两处必须同一种写法 —— 判决行和训练行要能互相对照, 换个格式就没法比了。 */
        unsigned char *keep = NULL;
        if (sft_rows) {
            keep = calloc((size_t)tk.S, 1);
            if (!keep) die("rows 掩码内存不足");
            long nk = 0;
            for (const char *q = sft_rows; *q; ) {
                int a, b, adv = 0;
                if (sscanf(q, "%d:%d%n", &a, &b, &adv) != 2) die("--rows 要 a:b 或 a:b,c:d,...(得到 \"%s\")", sft_rows);
                if (b > (int)tk.S) b = (int)tk.S;
                for (int r = a < 0 ? 0 : a; r < b; r++) { if (!keep[r]) nk++; keep[r] = 1; }
                q += adv;
                if (*q == ',') q++;
                else if (*q) die("--rows 格式错: %s", sft_rows);
            }
            printf("rows: 只把 %ld 行算进靶(共 %u 行)\n", nk, tk.S);
        }
        const int m = sft_build_target(&tk, wq8, onorm, Xall, xrows, 0, eta, X, R, keep);
        free(keep);
        if (m < 64) die("可用行只有 %d(topk %u 行 vs 捕获 %ld 行) — 对不上就是喂错了文件", m, tk.n, xrows);
        const int ev = m / 4 > 0 ? m / 4 : 1, fit = m - ev;
        double rn = 0;
        for (size_t i = 0; i < (size_t)m * DIM; i++) rn += (double)R[i] * R[i];
        printf("sft: 可用 %d 行 = fit %d + eval %d; 靶 RMS %.4g; rank %u λ %.4g η %.4g\n",
               m, fit, ev, sqrt(rn / ((double)m * DIM)), rank, lambda, eta);

        ds4_z *zl = ds4_z_solve(X, R, (uint32_t)fit, DIM, DIM, rank, lambda);
        if (!zl) die("sft 解算失败(退化)");
        double num = 0, den = 0;
        float *pred = calloc(DIM, sizeof(float));
        for (int r = fit; r < m; r++) {
            memset(pred, 0, DIM * sizeof(float));
            ds4_z_apply(zl, X + (size_t)r * DIM, pred);
            const float *rr = R + (size_t)r * DIM;
            for (uint32_t j = 0; j < DIM; j++) { const double d = rr[j] - pred[j]; num += d * d; den += (double)rr[j] * rr[j]; }
        }
        free(pred);
        const double gain = den > 0 ? 1.0 - num / den : 0.0;
        printf("  L%02d held-out 下降比 %.4f %s\n", sft_layer, gain, gain >= min_gain ? "✓ 注入" : "✗ 不注入(低于门槛)");
        /* ★held-out 只说"梯度方向能不能从 x 线性预测", 不等于 NLL 会降★ —— 真判决是
         * 挂上这个文件重跑 z_nightly_spark.sh nll, 看分歧段那个 +0.6595 有没有变小。 */
        if (gain < min_gain) die("L%02d 没过门槛(--min-gain %.3f) — 不产文件", sft_layer, min_gain);

        FILE *o2 = fopen(outp, "wb");
        if (!o2) die("输出打不开: %s", outp);
        uint32_t magic2 = 0x325A5144u, nl2 = 43u;
        fwrite(&magic2, 4, 1, o2); fwrite(&nl2, 4, 1, o2);
        for (uint32_t L = 0; L < nl2; L++) {
            if (L != (uint32_t)sft_layer) { uint32_t z = L, n0 = 0; fwrite(&z, 4, 1, o2); fwrite(&n0, 4, 1, o2); continue; }
            const uint32_t k = zl->k;
            const size_t nh = DS4_AMP_ZL_ELEMS(k, DIM, DIM);
            uint16_t *pay = malloc(nh * sizeof(uint16_t));
            if (!pay) die("载荷内存不足");
            size_t o = 0;
            for (uint32_t c = 0; c < k; c++) pay[o++] = ds4_f64_to_f16((double)zl->z[c]);
            for (size_t i = 0; i < (size_t)DIM * k; i++) pay[o++] = ds4_f64_to_f16((double)zl->U[i]);
            for (size_t i = 0; i < (size_t)DIM * k; i++) pay[o++] = ds4_f64_to_f16((double)zl->V[i]);
            uint32_t Lw = L, nops = 1, ty = 6u, psz = (uint32_t)(DS4_AMP_OP_HDR + nh * 2), din = DIM, dw = DIM;
            float tr = 1.0f;
            fwrite(&Lw, 4, 1, o2); fwrite(&nops, 4, 1, o2);
            fwrite(&ty, 4, 1, o2); fwrite(&psz, 4, 1, o2);
            fwrite(&k, 4, 1, o2); fwrite(&tr, 4, 1, o2);
            fwrite(&din, 4, 1, o2); fwrite(&dw, 4, 1, o2);
            fwrite(pay, 2, nh, o2);
            free(pay);
        }
        fclose(o2);
        printf("→ %s: L%02d 注入 rank %u(sft 梯度靶), held-out 下降比 %.4f\n", outp, sft_layer, zl->k, gain);
        ds4_z_free(zl); free(X); free(R); free(Xall);
        sft_topk_free(&tk); ds4_gguf_close(&gg);
        return 0;
    }

    int s0, nctx, body_lo, body_n;
    read_layout(lay, &s0, &nctx, &body_lo, &body_n);
    int L_lo = 0, L_hi = 42;
    if (lrange && sscanf(lrange, "%d-%d", &L_lo, &L_hi) != 2) die("--layers 要 a-b");

    /* 正文行: 教师侧 [body_lo, body_lo+body_n), 学生侧减去 nctx。fit/eval 按 3:1 切,
     * eval 取正文末段 —— 与反修的行切法同规矩(末 25% 留给打分)。 */
    const int n_body = body_n;
    const int n_ev = n_body / 4 > 0 ? n_body / 4 : 1;
    const int n_fit = n_body - n_ev;
    if (n_fit < 64) die("正文行太少(fit %d 行), 样本不够解 rank %u", n_fit, rank);
    printf("正文 %d 行 = fit %d + eval %d; 教师行 [%d,%d) ↔ 学生行 [%d,%d); rank %u λ %.4g\n",
           n_body, n_fit, n_ev, body_lo, body_lo + body_n,
           body_lo - nctx, body_lo - nctx + body_n, rank, lambda);

    FILE *out = fopen(outp, "wb");
    if (!out) die("输出打不开: %s", outp);
    uint32_t magic = 0x325A5144u, nl = 43u;
    fwrite(&magic, 4, 1, out); fwrite(&nl, 4, 1, out);

    int kept = 0;
    double gain_sum = 0;
    for (uint32_t L = 0; L < nl; L++) {
        if ((int)L < L_lo || (int)L > L_hi) {          /* 层范围外: 写空层, 保持 43 层结构 */
            uint32_t z = L, n0 = 0; fwrite(&z, 4, 1, out); fwrite(&n0, 4, 1, out);
            continue;
        }
        long tr_rows = 0, sr_rows = 0, so_rows = 0;
        const uint32_t DOUT = rte ? NEXP : DIM;
        const char *ynm = rte ? "raw_route_logits" : "raw_ffn_out";
        float *Yt = read_f16_rows(td, ynm, (int)L, &tr_rows, DOUT);
        float *Xs = read_f16_rows(sd, "raw_ffn_in",  (int)L, &sr_rows, DIM);
        float *Ys = read_f16_rows(sd, ynm, (int)L, &so_rows, DOUT);
        if (sr_rows != so_rows) die("L%u 学生两份捕获行数不一致(%ld vs %ld)", L, sr_rows, so_rows);
        if (body_lo + body_n > tr_rows) die("L%u 教师捕获 %ld 行 < 正文末 %d", L, tr_rows, body_lo + body_n);
        if (body_lo - nctx + body_n > sr_rows) die("L%u 学生捕获 %ld 行 < 正文末 %d", L, sr_rows, body_lo - nctx + body_n);

        float *X = malloc((size_t)n_body * DIM * sizeof(float));
        float *R = malloc((size_t)n_body * DOUT * sizeof(float));
        float *YS = malloc((size_t)n_body * DOUT * sizeof(float));   /* 学生原值(rte 判据要用) */
        float *YT = malloc((size_t)n_body * DOUT * sizeof(float));
        if (!X || !R || !YS || !YT) die("L%u 取料内存不足", L);
        for (int r = 0; r < n_body; r++) {
            const float *xs = Xs + (size_t)(body_lo - nctx + r) * DIM;
            const float *ys = Ys + (size_t)(body_lo - nctx + r) * DOUT;
            const float *yt = Yt + (size_t)(body_lo + r) * DOUT;
            float *xd = X + (size_t)r * DIM;
            for (uint32_t j = 0; j < DIM; j++) xd[j] = xs[j];
            float *rd = R + (size_t)r * DOUT, *sd2 = YS + (size_t)r * DOUT, *td2 = YT + (size_t)r * DOUT;
            for (uint32_t j = 0; j < DOUT; j++) { rd[j] = yt[j] - ys[j]; sd2[j] = ys[j]; td2[j] = yt[j]; }
        }
        free(Yt); free(Xs); free(Ys);

        /* 靶能量: 教师与学生差多少。★宽度用 DOUT★ —— rte 模式下 R 只有 256 宽,
         * 这里曾照抄 DIM(4096) 越界读 16 倍, 直接段错误(2026-09-08 实撞)。 */
        double rn = 0;
        for (size_t i = 0; i < (size_t)n_body * DOUT; i++) rn += (double)R[i] * R[i];

        ds4_z *zl = ds4_z_solve(X, R, (uint32_t)n_fit, DIM, DOUT, rank, lambda);
        if (!zl) { printf("  L%02u 解算失败(退化), 不注入\n", L);
                   uint32_t z = L, n0 = 0; fwrite(&z, 4, 1, out); fwrite(&n0, 4, 1, out);
                   free(X); free(R); free(YS); free(YT); continue; }

        /* ---- held-out 判据 ---- */
        double gain = 0;
        float s_scale = 1.0f;
        double ov0 = 0, ov1 = 0;
        if (rte) {
            /* 引擎算的是 δ[e]=Σ_c U[e][c]·z[c]·tanh((V[:,c]·x)/s), 解算解的是线性式。
             * 取 s = 4×p99|Vᵀx| 让 tanh 落在近线性段(|t|≤0.25 时 tanh(t)≈t 误差<2%),
             * 再把 1/s 折回 z ⇒ 回放出来就是解出来的那个线性映射。★评分必须用引擎的确切
             * 算式(含 tanh)回放★, 拿理想线性式打分等于给自己放水。 */
            const uint32_t k = zl->k;
            double *aa = malloc((size_t)n_fit * sizeof(double));
            double amax = 0;
            for (int r = 0; r < n_fit; r++) {
                const float *x = X + (size_t)r * DIM;
                double m = 0;
                for (uint32_t c = 0; c < k; c++) {
                    double v = 0;
                    for (uint32_t j = 0; j < DIM; j++) v += (double)zl->V[(size_t)j * k + c] * x[j];
                    if (fabs(v) > m) m = fabs(v);
                }
                aa[r] = m;
            }
            for (int r = 0; r < n_fit; r++) if (aa[r] > amax) amax = aa[r];
            free(aa);
            s_scale = (float)(4.0 * (amax > 0 ? amax : 1.0));
            /* 前 6 专家重合率: 加 δ 之前 vs 之后, 都跟教师的前 6 比 */
            double *pv = malloc((size_t)k * sizeof(double));
            float *lo = malloc(NEXP * sizeof(float));
            int nev2 = 0; double lnum = 0, lden = 0;
            for (int r = n_fit; r < n_body; r++) {
                const float *x = X + (size_t)r * DIM;
                for (uint32_t c = 0; c < k; c++) {
                    double v = 0;
                    for (uint32_t j = 0; j < DIM; j++) v += (double)zl->V[(size_t)j * k + c] * x[j];
                    pv[c] = tanh(v / s_scale) * (double)zl->z[c] * (double)s_scale;
                }
                const float *ys = YS + (size_t)r * NEXP, *yt = YT + (size_t)r * NEXP;
                for (uint32_t e = 0; e < NEXP; e++) {
                    double a = 0;
                    for (uint32_t c = 0; c < k; c++) a += pv[c] * (double)zl->U[(size_t)e * k + c];
                    lo[e] = ys[e] + (float)a;
                }
                ov0 += topk_overlap(ys, yt); ov1 += topk_overlap(lo, yt); nev2++;
                const float *rr = R + (size_t)r * NEXP;
                for (uint32_t e = 0; e < NEXP; e++) {           /* logits 残差降了多少 */
                    const double dd = (double)rr[e] - (lo[e] - ys[e]);
                    lnum += dd * dd; lden += (double)rr[e] * rr[e];
                }
            }
            free(pv); free(lo);
            ov0 /= nev2; ov1 /= nev2;
            gain = (ov1 - ov0) / (double)TOPK;      /* 重合率提升(占 6 个槽的比例) */
            printf("  L%02u 靶能量 %.4g | logits残差降 %.4f | 前6重合 %.3f→%.3f (+%.4f) | s=%.4g %s\n",
                   L, sqrt(rn / ((double)n_body * DOUT)),
                   lden > 0 ? 1.0 - lnum / lden : 0.0, ov0, ov1, gain, s_scale, gain >= min_gain ? "✓ 注入" : "✗ 不注入(低于门槛)");
        } else {
            double num = 0, den = 0;
            float *pred = calloc(DIM, sizeof(float));
            for (int r = n_fit; r < n_body; r++) {
                memset(pred, 0, DIM * sizeof(float));
                ds4_z_apply(zl, X + (size_t)r * DIM, pred);
                const float *rr = R + (size_t)r * DIM;
                for (uint32_t j = 0; j < DIM; j++) {
                    const double d = rr[j] - pred[j];
                    num += d * d; den += (double)rr[j] * rr[j];
                }
            }
            free(pred);
            gain = den > 0 ? 1.0 - num / den : 0.0;
            printf("  L%02u 靶能量 %.4g | held-out 下降比 %.4f %s\n",
                   L, sqrt(rn / ((double)n_body * DOUT)), gain,
                   gain >= min_gain ? "✓ 注入" : "✗ 不注入(低于门槛)");
        }
        free(X); free(R); free(YS); free(YT);
        if (gain < min_gain) {
            ds4_z_free(zl);
            uint32_t z = L, n0 = 0; fwrite(&z, 4, 1, out); fwrite(&n0, 4, 1, out);
            continue;
        }
        gain_sum += gain; kept++;

        /* type6 载荷: u32 k | f32 tr | u32 din | u32 dout | fp16 z[k],U[dout*k],V[din*k]。
         * tr=1.0: 夹持交给合并后的统一 clip(ds4_zfinetune.h "夹持语义变化"), 这里不再单独收紧。 */
        const uint32_t k = zl->k, dout = rte ? NEXP : DIM;
        const size_t nh = DS4_AMP_ZL_ELEMS(k, DIM, dout);
        uint16_t *pay = malloc(nh * sizeof(uint16_t));
        if (!pay) die("L%u 载荷内存不足", L);
        size_t o = 0;
        /* rte: z 里折进 s(引擎会算 tanh(a/s)·z, 近线性段等价于 z·a) */
        for (uint32_t c = 0; c < k; c++)
            pay[o++] = ds4_f64_to_f16((double)zl->z[c] * (rte ? (double)s_scale : 1.0));
        for (size_t i = 0; i < (size_t)dout * k; i++) pay[o++] = ds4_f64_to_f16((double)zl->U[i]);
        for (size_t i = 0; i < (size_t)DIM * k; i++) pay[o++] = ds4_f64_to_f16((double)zl->V[i]);
        uint32_t Lw = L, nops = 1, ty = rte ? 8u : 6u, psz = (uint32_t)(DS4_AMP_OP_HDR + nh * 2);
        float tr = rte ? s_scale : 1.0f;
        uint32_t din = DIM;
        fwrite(&Lw, 4, 1, out); fwrite(&nops, 4, 1, out);
        fwrite(&ty, 4, 1, out); fwrite(&psz, 4, 1, out);
        fwrite(&k, 4, 1, out); fwrite(&tr, 4, 1, out);
        fwrite(&din, 4, 1, out); { uint32_t dw = dout; fwrite(&dw, 4, 1, out); }
        fwrite(pay, 2, nh, out);
        free(pay);
        ds4_z_free(zl);
    }
    fclose(out);
    if (!kept) die("43 层没有一层过 held-out 门槛(--min-gain %.3f) — 今晚没学到东西, 不产文件", min_gain);
    printf("→ %s: %d/43 层注入(%s), 平均%s %.4f\n", outp, kept, mode,
           rte ? "前6重合提升" : "下降比", gain_sum / kept);
    return 0;
}
