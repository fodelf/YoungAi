/* zloss_solve.c — 反修唯一解算器(2026-08-26 重设计 v2, 用户两点理论落地)。
 *
 * ★口径: FP 自洽★ 教师与量化学生在同一个 FP 锚点取值:
 *   X = 锚 fin(FP 轨迹 x), 靶 R = zcache dH = Σw·Y_fp − Σw_q·Y_q (zlayer 模式①
 *   建的缓存, 冠军 r64c 已验证赢的口径); 学生输出 Ys = Σ pw·pYQ (zcache 自洽重建),
 *   教师 Yt = Ys + dH。v1 的混合口径(教师 FP 锚 × 学生引擎链捕获)把上游链漂移
 *   全塞进靶里, 深度单调致 36 层判空(裸 align L0 0.04→L27 0.60 即指纹), 已删除,
 *   与 XCAP 全量 0.47364 比裸差同罪(链态口径两次翻车)。
 *
 * ★动态 z(方案 B 兑现)★ 低维动态 z 映射高维行为: 锚 x 方向余弦 k-means 分 M 个
 *   模式(定死种子, LCG 首心 + farthest-point 续种, 全确定性), 每模式独立
 *   ds4_z_solve 闭式 rank-k, 应用时按最近质心选模式 —— 修正图随 token 的行为
 *   模式切换(路由哈希看似随机, 规律是条件性的; 一张全局静态图把模式平均掉,
 *   v1 深层全军判空的第二根因)。M=1 = 静态下界参赛。
 *
 * ★判空废除(铁律)★ 全网格输给裸 = 打印停车审计并以 exit 3 收尾(跑完所有层留全
 *   诊断表), 不写零混过去 —— "没找到规律"是解算器的 bug, 不是层的属性。
 *
 * 四损失 = ds4_loss 模块 held (M,λ,k) 网格择优; ER = held dH 能量挽回率。
 * 解算与运行时 apply 都是仓库根 ds4_z.c 同一份实现(反修/引擎复用铁律)。
 * 产物序列化(多模式容器/注入格式/判决尺升级闸)待针裁决后定, 本针只出诊断表。
 *
 * 用法: zloss_solve --anchor FILE --zcache DIR --out DIR --layers a-b
 *   [--ntok 8192] [--modes 1,8,16] [--ranks 16,64,128,256]
 *   [--lambdas 3e-3,3e-2,3e-1] [--wa 1] [--wc 0.5] [--ws 0.1] [--wf 1e-3]
 *   [--dither 0.04] [--seed 1] [--threads 16]
 *   [--fit-ranges a:b,..] [--ev-ranges a:b,..]   (行掩码: 拼接语料剔污染行)
 * 全 CLI 参数, 无环境变量(铁律)。
 */
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <pthread.h>
#include "npy.h"
#include "ds4_z.c"                      /* -I.. 仓库根: 与引擎同一份实现(复用铁律) */
#include "ds4_loss.c"

#define D 4096
#define MAXG 8                          /* λ/k/M 网格上限 */
#define MAXM 32                         /* 单档模式数上限 */

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "zloss_solve Error: "); vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n"); va_end(ap); exit(1);
}
static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) die("OOM %zu", n); return p; }

static int parse_ints(const char *s, int *out, int cap) {
    int n = 0; const char *p = s;
    while (*p && n < cap) { out[n++] = (int)strtol(p, (char **)&p, 10); if (*p == ',') p++; }
    return n;
}
static int parse_dbls(const char *s, double *out, int cap) {
    int n = 0; const char *p = s;
    while (*p && n < cap) { out[n++] = strtod(p, (char **)&p); if (*p == ',') p++; }
    return n;
}
static int *parse_ranges(const char *s, int *n_out) {   /* "a:b,c:d" 左闭右开 */
    int cap = 1024, n = 0; int *v = xmalloc((size_t)cap * sizeof(int));
    const char *p = s;
    while (*p) {
        char *e; long a = strtol(p, &e, 10);
        if (*e != ':') die("区间语法(a:b): %s", s);
        long b = strtol(e + 1, &e, 10);
        for (long i = a; i < b; i++) {
            if (n == cap) { cap *= 2; v = realloc(v, (size_t)cap * sizeof(int)); if (!v) die("OOM"); }
            v[n++] = (int)i;
        }
        p = (*e == ',') ? e + 1 : e;
    }
    *n_out = n; return v;
}

/* ---- 锚 fin 读取(DQA2, 与 zlayer anchor_layer 同格式; 只取 fin, 路由不用) ---- */
static float *anchor_fin(const char *ap, int L, int ntok) {
    FILE *f = fopen(ap, "rb"); if (!f) die("锚打不开: %s", ap);
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8) die("锚头截断: %s", ap);
    if (hd[0] != 0x32415144u) die("%s 不是 DQA2 锚(magic 0x%08x)", ap, hd[0]);
    int S = (int)hd[1], DIM = (int)hd[3], NL = (int)hd[4];
    if (DIM != D) die("锚 DIM=%d ≠ %d", DIM, D);
    if (ntok > S) die("--ntok %d > 锚 S=%d", ntok, S);
    if (L < 0 || L >= NL) die("L%d 越界(锚 NL=%d)", L, NL);
    float *fin = xmalloc((size_t)ntok * D * sizeof(float));
    if (fseeko(f, (off_t)(40LL + (long long)L * S * DIM * 4), SEEK_SET) ||
        fread(fin, 4, (size_t)ntok * D, f) != (size_t)ntok * D) die("锚 fin 读不满 L=%d", L);
    fclose(f); return fin;
}

/* ---- zcache 读取: R=dH, Ys=Σ pw·pYQ(自洽重建, 不掺 teacher_routed 的约定差) ---- */
static void zcache_load(const char *dir, int L, int ntok, float **R_out, float **Ys_out) {
    char p[1024]; snprintf(p, sizeof p, "%s/zcache_L%02d.npz", dir, L);
    FILE *f = fopen(p, "rb");
    if (!f) die("%s 打不开(先 DS4_ZL_CACHE_ONLY=1 跑 zlayer 建缓存)", p);
    fseeko(f, 0, SEEK_END); long long sz = (long long)ftello(f); fseeko(f, 0, SEEK_SET);
    uint8_t *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("%s 读不满", p);
    fclose(f);
    npz_arr dH, prow, pe, pw, pYQ;
    if (npz_get(buf, sz, "dH", &dH) || npz_get(buf, sz, "prow", &prow) ||
        npz_get(buf, sz, "pe", &pe) || npz_get(buf, sz, "pw", &pw) ||
        npz_get(buf, sz, "pYQ", &pYQ)) die("%s 字段缺(要 dH/prow/pe/pw/pYQ)", p);
    if (dH.d0 < ntok || dH.d1 != D)
        die("zcache dH 形状 %lldx%lld, 需 ≥%dx%d", (long long)dH.d0, (long long)dH.d1, ntok, D);
    if (pYQ.d1 != D || pYQ.d0 != prow.n || pw.n != prow.n || pe.n != prow.n)
        die("zcache 配对形状不一致: prow=%lld pYQ=%lldx%lld", (long long)prow.n,
            (long long)pYQ.d0, (long long)pYQ.d1);
    float *R = xmalloc((size_t)ntok * D * sizeof(float));
    for (size_t i = 0; i < (size_t)ntok * D; i++) R[i] = (float)dH.v[i];
    float *Ys = xmalloc((size_t)ntok * D * sizeof(float));
    memset(Ys, 0, (size_t)ntok * D * sizeof(float));
    for (long long i = 0; i < prow.n; i++) {
        int t = (int)prow.v[i];
        if (t < 0 || (int)pe.v[i] < 0) die("zcache 配对越界: prow=%d pe=%d", t, (int)pe.v[i]);
        if (t >= ntok) continue;
        float w = (float)pw.v[i];
        float *ys = Ys + (size_t)t * D;
        const double *yq = pYQ.v + (size_t)i * D;
        for (int d = 0; d < D; d++) ys[d] += w * (float)yq[d];
    }
    free(dH.v); free(prow.v); free(pe.v); free(pw.v); free(pYQ.v); free(buf);
    *R_out = R; *Ys_out = Ys;
}

/* ---- 方向余弦 k-means(全确定性): LCG 首心 + farthest-point 续种 + 20 轮 Lloyd ---- */
static int mode_assign(const float *x, const float *C, int M) {
    if (M <= 1) return 0;
    int bm = 0; double bd = -1e30;
    for (int m = 0; m < M; m++) {
        const float *c = C + (size_t)m * D;
        double d = 0; for (int j = 0; j < D; j++) d += (double)x[j] * c[j];
        if (d > bd) { bd = d; bm = m; }
    }
    return bm;
}
static void kmeans_modes(const float *X, const int *fit, int nf, int M, uint64_t seed, float *C) {
    float *Xn = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        const float *x = X + (size_t)fit[i] * D; float *o = Xn + (size_t)i * D;
        double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
        double inv = ss > 0 ? 1.0 / sqrt(ss) : 0.0;
        for (int j = 0; j < D; j++) o[j] = (float)(x[j] * inv);
    }
    uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    memcpy(C, Xn + (size_t)((s >> 33) % (uint64_t)nf) * D, D * sizeof(float));
    float *dmin = xmalloc((size_t)nf * sizeof(float));
    for (int i = 0; i < nf; i++) dmin[i] = 1e30f;
    for (int m = 1; m < M; m++) {
        int far = 0; float fd = -1e30f;
        for (int i = 0; i < nf; i++) {
            const float *c = C + (size_t)(m - 1) * D, *x = Xn + (size_t)i * D;
            double dt = 0; for (int j = 0; j < D; j++) dt += (double)x[j] * c[j];
            float d = 1.0f - (float)dt;
            if (d < dmin[i]) dmin[i] = d;
            if (dmin[i] > fd) { fd = dmin[i]; far = i; }
        }
        memcpy(C + (size_t)m * D, Xn + (size_t)far * D, D * sizeof(float));
        dmin[far] = -1e30f;
    }
    int *asg = xmalloc((size_t)nf * sizeof(int));
    double *acc = xmalloc((size_t)M * D * sizeof(double));
    int *cnt = xmalloc((size_t)M * sizeof(int));
    for (int it = 0; it < 20; it++) {
        for (int i = 0; i < nf; i++) asg[i] = mode_assign(Xn + (size_t)i * D, C, M);
        memset(acc, 0, (size_t)M * D * sizeof(double));
        memset(cnt, 0, (size_t)M * sizeof(int));
        for (int i = 0; i < nf; i++) {
            double *a = acc + (size_t)asg[i] * D; const float *x = Xn + (size_t)i * D;
            for (int j = 0; j < D; j++) a[j] += x[j];
            cnt[asg[i]]++;
        }
        for (int m = 0; m < M; m++) {
            if (!cnt[m]) {                       /* 空簇: 重播到当前最不合群的行 */
                int worst = 0; double wd = 1e30;
                for (int i = 0; i < nf; i++) {
                    const float *x = Xn + (size_t)i * D, *c = C + (size_t)asg[i] * D;
                    double dt = 0; for (int j = 0; j < D; j++) dt += (double)x[j] * c[j];
                    if (dt < wd) { wd = dt; worst = i; }
                }
                memcpy(C + (size_t)m * D, Xn + (size_t)worst * D, D * sizeof(float));
                continue;
            }
            double ss = 0; const double *a = acc + (size_t)m * D;
            for (int j = 0; j < D; j++) ss += a[j] * a[j];
            double inv = ss > 0 ? 1.0 / sqrt(ss) : 0.0;
            float *c = C + (size_t)m * D;
            for (int j = 0; j < D; j++) c[j] = (float)(a[j] * inv);
        }
    }
    free(Xn); free(dmin); free(asg); free(acc); free(cnt);
}

/* ---- held 行评估(pthread): Yhat=Ys+M(x), Cb=M(x), Cp=M(x+δ); ER 部分和 ---- */
typedef struct {
    ds4_z *const *zl; const float *C; int M;
    const float *X, *Ys, *R; const int *ev, *mode_ev; int nev;
    float *Yhat, *Cb, *Cp; double dscale; uint64_t seed;
    double er_num, er_den;
    int t0, t1;
} ev_ctx;
static void *ev_worker(void *arg) {
    ev_ctx *c = (ev_ctx *)arg;
    float *delta = xmalloc(D * sizeof(float));
    float *xp = xmalloc(D * sizeof(float));
    c->er_num = c->er_den = 0;
    for (int i = c->t0; i < c->t1; i++) {
        const float *x = c->X + (size_t)c->ev[i] * D;
        float *yh = c->Yhat + (size_t)i * D, *cb = c->Cb + (size_t)i * D, *cp = c->Cp + (size_t)i * D;
        memcpy(yh, c->Ys + (size_t)c->ev[i] * D, D * sizeof(float));
        memset(cb, 0, D * sizeof(float)); memset(cp, 0, D * sizeof(float));
        int m = c->mode_ev[i];
        if (c->zl[m]) { ds4_z_apply(c->zl[m], x, yh); ds4_z_apply(c->zl[m], x, cb); }
        double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
        const float rms = (float)sqrt(ss / D);
        ds4_loss_dither(c->seed, (uint32_t)c->ev[i], (float)(c->dscale * rms), delta, D);
        for (int j = 0; j < D; j++) xp[j] = x[j] + delta[j];
        int mp = mode_assign(xp, c->C, c->M);    /* 扰动可换模式: 门稳定性一并入 smooth */
        if (c->zl[mp]) ds4_z_apply(c->zl[mp], xp, cp);
        const float *r = c->R + (size_t)c->ev[i] * D;
        for (int j = 0; j < D; j++) {
            double e = (double)r[j] - cb[j];
            c->er_num += e * e; c->er_den += (double)r[j] * r[j];
        }
    }
    free(delta); free(xp); return NULL;
}
static double eval_apply(ds4_z *const *zl, const float *C, int M, const float *X,
                         const float *Ys, const float *R, const int *ev, const int *mode_ev,
                         int nev, float *Yhat, float *Cb, float *Cp,
                         double dscale, uint64_t seed, int nth) {
    pthread_t th[32]; ev_ctx cx[32];
    if (nth > 32) nth = 32;
    for (int t = 0; t < nth; t++) {
        cx[t] = (ev_ctx){zl, C, M, X, Ys, R, ev, mode_ev, nev, Yhat, Cb, Cp, dscale, seed,
                         0, 0, nev * t / nth, nev * (t + 1) / nth};
        pthread_create(&th[t], NULL, ev_worker, &cx[t]);
    }
    double num = 0, den = 0;
    for (int t = 0; t < nth; t++) { pthread_join(th[t], NULL); num += cx[t].er_num; den += cx[t].er_den; }
    return den > 0 ? 1.0 - num / den : 0.0;      /* ER: held dH 能量挽回率 */
}

int main(int argc, char **argv) {
    const char *anc = NULL, *zdir = NULL, *out = NULL, *frs = NULL, *ers = NULL;
    int l0 = -1, l1 = -1, ntok = 8192, nfit = 6144, nth = 16;
    int modes[MAXG] = {1, 8, 16}; int nmode = 3;
    int ranks[MAXG] = {16, 64, 128, 256}; int nrank = 4;
    double lambdas[MAXG] = {3e-3, 3e-2, 3e-1}; int nlam = 3;
    ds4_loss_weights lw = {1.0f, 0.5f, 0.1f, 1e-3f};
    double dscale = 0.04; uint64_t seed = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--anchor") && i + 1 < argc) anc = argv[++i];
        else if (!strcmp(argv[i], "--zcache") && i + 1 < argc) zdir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d-%d", &l0, &l1) != 2) die("--layers a-b");
        }
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nfit") && i + 1 < argc) nfit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--modes") && i + 1 < argc) nmode = parse_ints(argv[++i], modes, MAXG);
        else if (!strcmp(argv[i], "--ranks") && i + 1 < argc) nrank = parse_ints(argv[++i], ranks, MAXG);
        else if (!strcmp(argv[i], "--lambdas") && i + 1 < argc) nlam = parse_dbls(argv[++i], lambdas, MAXG);
        else if (!strcmp(argv[i], "--wa") && i + 1 < argc) lw.w_align = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--wc") && i + 1 < argc) lw.w_classify = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--ws") && i + 1 < argc) lw.w_smooth = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--wf") && i + 1 < argc) lw.w_fixed = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--dither") && i + 1 < argc) dscale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--fit-ranges") && i + 1 < argc) frs = argv[++i];
        else if (!strcmp(argv[i], "--ev-ranges") && i + 1 < argc) ers = argv[++i];
        else die("未知参数 %s", argv[i]);
    }
    if (!anc || !zdir || !out || l0 < 0)
        die("用法: zloss_solve --anchor FILE --zcache DIR --out DIR --layers a-b ...(见文件头)");

    int nf, nev, *fit, *ev;
    if (frs) {
        if (!ers) die("--fit-ranges 设了必须同时给 --ev-ranges");
        fit = parse_ranges(frs, &nf); ev = parse_ranges(ers, &nev);
    } else {
        nf = nfit; nev = ntok - nfit;
        fit = xmalloc((size_t)nf * sizeof(int)); ev = xmalloc((size_t)nev * sizeof(int));
        for (int i = 0; i < nf; i++) fit[i] = i;
        for (int i = 0; i < nev; i++) ev[i] = nfit + i;
    }
    for (int i = 0; i < nf; i++) if (fit[i] >= ntok) die("fit 行 %d ≥ ntok", fit[i]);
    for (int i = 0; i < nev; i++) if (ev[i] >= ntok) die("ev 行 %d ≥ ntok", ev[i]);
    int maxk = 0, maxM = 0;
    for (int i = 0; i < nrank; i++) if (ranks[i] > maxk) maxk = ranks[i];
    for (int i = 0; i < nmode; i++) { if (modes[i] > maxM) maxM = modes[i]; if (modes[i] > MAXM) die("M>%d", MAXM); }
    printf("zloss_solve v2(FP自洽+动态z) L%d-%d fit=%d ev=%d M=%d档 k=%d档(max %d) λ=%d档 "
           "w=[%.3g %.3g %.3g %.3g] dither=%.3g seed=%llu\n",
           l0, l1, nf, nev, nmode, nrank, maxk, nlam,
           lw.w_align, lw.w_classify, lw.w_smooth, lw.w_fixed, dscale, (unsigned long long)seed);

    int any_lost = 0;
    for (int L = l0; L <= l1; L++) {
        float *X = anchor_fin(anc, L, ntok);
        float *R = NULL, *Ys = NULL;
        zcache_load(zdir, L, ntok, &R, &Ys);
        float *Yt = xmalloc((size_t)ntok * D * sizeof(float));
        for (size_t i = 0; i < (size_t)ntok * D; i++) Yt[i] = Ys[i] + R[i];

        float *Yt_ev = xmalloc((size_t)nev * D * sizeof(float));
        float *Ys_ev = xmalloc((size_t)nev * D * sizeof(float));
        for (int i = 0; i < nev; i++) {
            memcpy(Yt_ev + (size_t)i * D, Yt + (size_t)ev[i] * D, D * sizeof(float));
            memcpy(Ys_ev + (size_t)i * D, Ys + (size_t)ev[i] * D, D * sizeof(float));
        }
        float *wv = xmalloc(D * sizeof(float));
        ds4_loss_dim_variance(Yt_ev, (uint32_t)nev, D, wv);
        float la0 = ds4_loss_align(Ys_ev, Yt_ev, (uint32_t)nev, D);
        float lc0 = ds4_loss_classify(Ys_ev, Yt_ev, wv, (uint32_t)nev, D);
        float tot0 = ds4_loss_total(&lw, la0, lc0, 0.0f, 0.0f);

        char lossp[1024]; snprintf(lossp, sizeof lossp, "%s/fourloss_L%02d.txt", out, L);
        FILE *lf = fopen(lossp, "w"); if (!lf) die("%s 写不开", lossp);
        fprintf(lf, "# zloss v2 L%02d 裸基线: align=%.6f cls=%.6f total=%.6f\n", L, la0, lc0, tot0);
        fprintf(lf, "# M lambda k align classify smooth fixed total ER%% volMB\n");
        printf("★L%d 裸: align %.4f cls %.4f total %.4f\n", L, la0, lc0, tot0);

        float *Yhat = xmalloc((size_t)nev * D * sizeof(float));
        float *Cb = xmalloc((size_t)nev * D * sizeof(float));
        float *Cp = xmalloc((size_t)nev * D * sizeof(float));
        float *zcat = xmalloc((size_t)maxM * maxk * sizeof(float));
        double best_tot = tot0, best_lam = 0, best_er = 0; int best_M = 0, best_k = 0;
        float bla = la0, blc = lc0;

        for (int mi = 0; mi < nmode; mi++) {
            const int M = modes[mi];
            float *C = xmalloc((size_t)M * D * sizeof(float));
            if (M > 1) kmeans_modes(X, fit, nf, M, seed, C);
            else { memset(C, 0, (size_t)D * sizeof(float)); }
            int *mode_fit = xmalloc((size_t)nf * sizeof(int));
            int *mode_ev = xmalloc((size_t)nev * sizeof(int));
            int cnt[MAXM]; memset(cnt, 0, sizeof cnt);
            for (int i = 0; i < nf; i++) { mode_fit[i] = mode_assign(X + (size_t)fit[i] * D, C, M); cnt[mode_fit[i]]++; }
            for (int i = 0; i < nev; i++) mode_ev[i] = mode_assign(X + (size_t)ev[i] * D, C, M);
            printf("  L%d M=%d fit行分布:", L, M);
            for (int m = 0; m < M; m++) printf(" %d", cnt[m]);
            printf("\n");

            for (int li = 0; li < nlam; li++) {
                ds4_z *zl[MAXM] = {0};
                for (int m = 0; m < M; m++) {
                    if (!cnt[m]) { printf("  L%d M=%d λ=%g 模式%d 无fit行, 该模式不修\n", L, M, lambdas[li], m); continue; }
                    float *Xm = xmalloc((size_t)cnt[m] * D * sizeof(float));
                    float *Rm = xmalloc((size_t)cnt[m] * D * sizeof(float));
                    int w = 0;
                    for (int i = 0; i < nf; i++) if (mode_fit[i] == m) {
                        memcpy(Xm + (size_t)w * D, X + (size_t)fit[i] * D, D * sizeof(float));
                        memcpy(Rm + (size_t)w * D, R + (size_t)fit[i] * D, D * sizeof(float));
                        w++;
                    }
                    int rk = maxk < cnt[m] ? maxk : cnt[m];
                    zl[m] = ds4_z_solve(Xm, Rm, (uint32_t)cnt[m], D, D, (uint32_t)rk, (float)lambdas[li]);
                    if (!zl[m]) die("L%d M=%d λ=%g 模式%d 解算失败", L, M, lambdas[li], m);
                    free(Xm); free(Rm);
                }
                for (int ki = 0; ki < nrank; ki++) {
                    long long vol = (long long)M * D * 4;    /* 质心 f32 也计体积 */
                    int nz = 0;
                    for (int m = 0; m < M; m++) {
                        if (!zl[m]) continue;
                        ds4_z_set_rank(zl[m], (uint32_t)ranks[ki]);
                        vol += (long long)zl[m]->k * (1 + 2 * D) * 2;   /* fp16 z|U|V */
                        for (uint32_t c = 0; c < zl[m]->k; c++) zcat[nz++] = zl[m]->z[c];
                    }
                    double er = eval_apply(zl, C, M, X, Ys, R, ev, mode_ev, nev,
                                           Yhat, Cb, Cp, dscale, seed, nth);
                    float la = ds4_loss_align(Yhat, Yt_ev, (uint32_t)nev, D);
                    float lc = ds4_loss_classify(Yhat, Yt_ev, wv, (uint32_t)nev, D);
                    float ls = ds4_loss_smooth(Cb, Cp, (uint32_t)nev, D);
                    float lfx = ds4_loss_fixed(zcat, (size_t)nz);
                    float tot = ds4_loss_total(&lw, la, lc, ls, lfx);
                    fprintf(lf, "%d %.3g %d %.6f %.6f %.6f %.6f %.6f %.2f %.2f\n",
                            M, lambdas[li], ranks[ki], la, lc, ls, lfx, tot, er * 100, vol / 1e6);
                    printf("  L%d M=%-2d λ=%-5.3g k=%-3d | align %.4f cls %.4f sm %.5f fx %.5f "
                           "tot %.4f | ER %.1f%% | vol %.1fMB\n",
                           L, M, lambdas[li], ranks[ki], la, lc, ls, lfx, tot, er * 100, vol / 1e6);
                    if (tot < best_tot) {
                        best_tot = tot; best_lam = lambdas[li]; best_k = ranks[ki];
                        best_M = M; best_er = er; bla = la; blc = lc;
                    }
                    fflush(stdout);
                }
                for (int m = 0; m < M; m++) if (zl[m]) ds4_z_free(zl[m]);
            }
            free(C); free(mode_fit); free(mode_ev);
        }
        int won = best_M > 0;
        fprintf(lf, "# 选中: %s M=%d λ=%.3g k=%d total=%.6f (裸 %.6f Δ=%.2f%%) ER=%.1f%%\n",
                won ? "有解" : "全网格输裸(停车审计)", best_M, best_lam, best_k,
                best_tot, tot0, tot0 > 0 ? 100.0 * (tot0 - best_tot) / tot0 : 0.0, best_er * 100);
        fclose(lf);
        printf("★L%d 终判: %s M=%d λ=%.3g k=%d | align %.4f→%.4f cls %.4f→%.4f "
               "total %.4f→%.4f | ER %.1f%%\n",
               L, won ? "选中" : "★全网格输裸=解算器有病, 停车审计★", best_M, best_lam,
               best_k, la0, bla, lc0, blc, tot0, best_tot, best_er * 100);
        if (!won) any_lost = 1;
        fflush(stdout);
        free(X); free(R); free(Ys); free(Yt); free(Yt_ev); free(Ys_ev); free(wv);
        free(Yhat); free(Cb); free(Cp); free(zcat);
    }
    free(fit); free(ev);
    if (any_lost) { fprintf(stderr, "zloss_solve: 有层全网格输裸 — 停车审计(exit 3)\n"); return 3; }
    return 0;
}
