/* zloss_solve.c — 反修唯一解算器(2026-08-26 重设计 v2, 用户两点理论落地)。
 *
 * ★口径: FP 自洽★ 教师与量化学生在同一个 FP 锚点取值:
 *   X = 锚 fin(FP 轨迹 x), 靶 R = zcache dH = Σw·Y_fp − Σw_q·Y_q (zlayer 模式①
 *   建的缓存, 冠军 r64c 已验证赢的口径); 学生输出 Ys = Σ pw·pYQ (zcache 自洽重建),
 *   教师 Yt = Ys + dH。v1 的混合口径(教师 FP 锚 × 学生引擎链捕获)把上游链漂移
 *   全塞进靶里, 深度单调致 36 层判空(裸 align L0 0.04→L27 0.60 即指纹), 已删除,
 *   与 XCAP 全量 0.47364 比裸差同罪(链态口径两次翻车)。
 *
 * ★条件化(用户理论)★ 修正按离散路由结构切换: cond 臂=top1 专家分组 g_e+b_e
 *   (2026-07-05 真尺 +9.0pp; 同期全局线性 z rank16 = −5915% 判"函数形态错")。
 *   EM 模式门/x 门为历史对照臂(门读不出标签, 三层同款结论)。
 *
 * ★判空废除(铁律)★ 全网格输给裸 = 打印停车审计并以 exit 3 收尾(跑完所有层留全
 *   诊断表), 不写零混过去 —— "没找到规律"是解算器的 bug, 不是层的属性。
 *
 * 四损失 = ds4_loss 模块 held (M,λ,k) 网格择优; ER = held dH 能量挽回率。
 * 解算与运行时 apply 都是仓库根 ds4_z.c 同一份实现(反修/引擎复用铁律)。
 * 产物序列化(多模式容器/注入格式/判决尺升级闸)待针裁决后定, 本针只出诊断表。
 *
 * 用法: zloss_solve --anchor FILE --zcache DIR --out DIR --layers a-b
 *   [--ntok 8192] [--modes 1,8,16] [--ranks 16,64,128,256] [--lambdas 3e-3,3e-2,3e-1]
 *   [--wa 1] [--wc 0.5] [--ws 0.1] [--wf 1e-3] [--dither 0.04] [--seed 1] [--threads 16]
 *   [--fit-ranges a:b,..] [--ev-ranges a:b,..] 行掩码 | [--emit-ge/-z DIR] 落地路
 * 全 CLI 参数, 无环境变量(铁律)。
 */
#define _FILE_OFFSET_BITS 64
#include <time.h>
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
#include "linalg_small.h"               /* GE 256×256 正规方程用 chol_solve_spd */

#define D 4096
#define MAXG 8                          /* λ/k/M 网格上限 */
#define MAXM 32                         /* 单档模式数上限 */

static double tnow(void) {              /* 相位计时(提速轮的肥肉探测器) */
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "zloss_solve Error: "); vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n"); va_end(ap); exit(1);
}
static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) die("OOM %zu", n); return p; }

static int g_smooth_aug = 0;            /* --smooth-aug: L_smooth 进解算目标(扰动增广) */
static int g_track_z = 0;               /* --emit-z: 网格中跟踪冠军 z 克隆(序列化用) */
static int g_dynz = 0;                  /* --dynz: 方案B 动态 z 臂(用户核心设计) */

#include "zloss_arms.inc.c"             /* 解算臂分片(金标+GE/ftA臂+模式发现+zcache) */

/* ---- 锚读取(DQA2, 与 zlayer anchor_layer 同格式): fin + top-1 专家(路由门用,
 * rw 最大槽位; 部署时路由先于专家计算, 是免费可得的信号) ---- */
static float *anchor_load(const char *ap, int L, int ntok, int **top1_out) {
    FILE *f = fopen(ap, "rb"); if (!f) die("锚打不开: %s", ap);
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8) die("锚头截断: %s", ap);
    if (hd[0] != 0x32415144u) die("%s 不是 DQA2 锚(magic 0x%08x)", ap, hd[0]);
    int S = (int)hd[1], DIM = (int)hd[3], NL = (int)hd[4], NACT = (int)hd[6];
    if (DIM != D) die("锚 DIM=%d ≠ %d", DIM, D);
    if (ntok > S) die("--ntok %d > 锚 S=%d", ntok, S);
    if (L < 0 || L >= NL) die("L%d 越界(锚 NL=%d)", L, NL);
    if (NACT < 1 || NACT > 16) die("锚 NACT=%d 口径可疑", NACT);
    float *fin = xmalloc((size_t)ntok * D * sizeof(float));
    if (fseeko(f, (off_t)(40LL + (long long)L * S * DIM * 4), SEEK_SET) ||
        fread(fin, 4, (size_t)ntok * D, f) != (size_t)ntok * D) die("锚 fin 读不满 L=%d", L);
    long long ridx_off = 40LL + (long long)NL * S * DIM * 4;
    long long rw_off = ridx_off + (long long)NL * S * NACT * 4;
    int *ridx = xmalloc((size_t)ntok * NACT * sizeof(int));
    float *rw = xmalloc((size_t)ntok * NACT * sizeof(float));
    if (fseeko(f, (off_t)(ridx_off + (long long)L * S * NACT * 4), SEEK_SET) ||
        fread(ridx, 4, (size_t)ntok * NACT, f) != (size_t)ntok * NACT) die("锚 ridx 读不满 L=%d", L);
    if (fseeko(f, (off_t)(rw_off + (long long)L * S * NACT * 4), SEEK_SET) ||
        fread(rw, 4, (size_t)ntok * NACT, f) != (size_t)ntok * NACT) die("锚 rw 读不满 L=%d", L);
    fclose(f);
    int *top1 = xmalloc((size_t)ntok * sizeof(int));
    for (int t = 0; t < ntok; t++) {
        int bs = 0;
        for (int s = 1; s < NACT; s++) if (rw[(size_t)t * NACT + s] > rw[(size_t)t * NACT + bs]) bs = s;
        int e = ridx[(size_t)t * NACT + bs];
        if (e < 0 || e > 255) die("锚 ridx 越界: %d", e);
        top1[t] = e;
    }
    free(ridx); free(rw);
    *top1_out = top1; return fin;
}


/* 图应用: d_in=3D 的 ftA 图先做 φ 提升(与引擎 type6 ftA 路同式) */
static void apply_any(const ds4_z *zl, const float *x, float *phib, float *y) {
    if (zl->d_in == 3 * D) { mk_phi(x, phib); ds4_z_apply(zl, phib, y); }
    else ds4_z_apply(zl, x, y);
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
#include "zloss_gate.inc.c"             /* 门族分片: EM 模式门 + 语境门 GEc */
#include "zloss_emit.inc.c"             /* GE-only 端到端注入(bf.GE, 斜率标定) */
#include "zloss_dynz.inc.c"   /* 动态 z 族(方案B/正交靶/4L) */
#include "zloss_cond.inc.c"   /* 条件化 z: top1 专家分组 g_e+b_e(+9.0pp 配方) */

/* ---- held 行评估(pthread): Yhat=Ys+M(x), Cb=M(x), Cp=M(x+δ); ER 部分和 ---- */
typedef struct {
    ds4_z *const *zl; const float *C; int M, gate_route;
    const float *bc;                    /* 基修正(ntok×D, GE 臂), NULL=无 */
    float tr;                           /* 信任域(引擎 zc_zl_add_kernel 同式), ≤0=不夹 */
    const float *X, *Ys, *R; const int *ev, *mode_ev; int nev;
    float *Yhat, *Cb, *Cp; double dscale; uint64_t seed;
    double er_num, er_den;
    int t0, t1;
} ev_ctx;

/* 图输出按引擎信任域夹持后叠加: cap=tr·‖基‖, ‖Δ‖>cap 整体缩到 cap。
 * 与部署 apply 同式 —— 评估不建模夹持=评一个不存在的部署行为。 */
static void add_clamped(const float *mb, float nb, float tr, float *yh, float *cb) {
    double nd2 = 0;
    for (int j = 0; j < D; j++) nd2 += (double)mb[j] * mb[j];
    float s = 1.0f;
    if (tr > 0 && nb > 0) {
        float nd = (float)sqrt(nd2), cap = tr * nb;
        if (nd > cap && nd > 0) s = cap / nd;
    }
    for (int j = 0; j < D; j++) {
        if (yh) yh[j] += s * mb[j];
        if (cb) cb[j] += s * mb[j];
    }
}
static void *ev_worker(void *arg) {
    ev_ctx *c = (ev_ctx *)arg;
    float *delta = xmalloc(D * sizeof(float));
    float *xp = xmalloc(D * sizeof(float));
    float *phib = xmalloc((size_t)3 * D * sizeof(float));
    float *mbuf = xmalloc(D * sizeof(float));
    c->er_num = c->er_den = 0;
    for (int i = c->t0; i < c->t1; i++) {
        const float *x = c->X + (size_t)c->ev[i] * D;
        float *yh = c->Yhat + (size_t)i * D, *cb = c->Cb + (size_t)i * D, *cp = c->Cp + (size_t)i * D;
        memcpy(yh, c->Ys + (size_t)c->ev[i] * D, D * sizeof(float));
        if (c->bc) {                     /* 基修正(GE): 图叠在它之上 */
            const float *b = c->bc + (size_t)c->ev[i] * D;
            for (int j = 0; j < D; j++) yh[j] += b[j];
            memcpy(cb, b, D * sizeof(float)); memcpy(cp, b, D * sizeof(float));
        } else { memset(cb, 0, D * sizeof(float)); memset(cp, 0, D * sizeof(float)); }
        int m = c->mode_ev[i];
        double nb2 = 0;                  /* ‖基‖=Ys(+GE) 行范数, 引擎 nr 同义 */
        for (int j = 0; j < D; j++) nb2 += (double)yh[j] * yh[j];
        const float nb = (float)sqrt(nb2);
        if (c->zl[m]) {
            memset(mbuf, 0, D * sizeof(float));
            apply_any(c->zl[m], x, phib, mbuf);
            add_clamped(mbuf, nb, c->tr, yh, cb);
        }
        double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
        const float rms = (float)sqrt(ss / D);
        ds4_loss_dither(c->seed, (uint32_t)c->ev[i], (float)(c->dscale * rms), delta, D);
        for (int j = 0; j < D; j++) xp[j] = x[j] + delta[j];
        /* 扰动可换模式(x门): 门稳定性一并入 smooth; 路由门下扰动不改路由 → 同模式 */
        int mp = c->gate_route ? m : mode_assign(xp, c->C, c->M);
        if (c->zl[mp]) {
            memset(mbuf, 0, D * sizeof(float));
            apply_any(c->zl[mp], xp, phib, mbuf);
            add_clamped(mbuf, nb, c->tr, cp, NULL);
        }
        const float *r = c->R + (size_t)c->ev[i] * D;
        for (int j = 0; j < D; j++) {
            double e = (double)r[j] - cb[j];
            c->er_num += e * e; c->er_den += (double)r[j] * r[j];
        }
    }
    free(delta); free(xp); free(phib); free(mbuf); return NULL;
}
static double eval_apply(ds4_z *const *zl, const float *C, int M, int gate_route,
                         const float *bc, float tr, const float *X,
                         const float *Ys, const float *R, const int *ev, const int *mode_ev,
                         int nev, float *Yhat, float *Cb, float *Cp,
                         double dscale, uint64_t seed, int nth) {
    pthread_t th[32]; ev_ctx cx[32];
    if (nth > 32) nth = 32;
    for (int t = 0; t < nth; t++) {
        cx[t] = (ev_ctx){zl, C, M, gate_route, bc, tr, X, Ys, R, ev, mode_ev, nev, Yhat, Cb, Cp,
                         dscale, seed, 0, 0, nev * t / nth, nev * (t + 1) / nth};
        pthread_create(&th[t], NULL, ev_worker, &cx[t]);
    }
    double num = 0, den = 0;
    for (int t = 0; t < nth; t++) { pthread_join(th[t], NULL); num += cx[t].er_num; den += cx[t].er_den; }
    return den > 0 ? 1.0 - num / den : 0.0;      /* ER: held dH 能量挽回率 */
}

int main(int argc, char **argv) {
    const char *anc = NULL, *zdir = NULL, *out = NULL, *frs = NULL, *ers = NULL,
               *emitge = NULL, *emitz = NULL;
    int l0 = -1, l1 = -1, ntok = 8192, nfit = 6144, nth = 16, selftest = 0;
    int modes[MAXG] = {1, 8, 16}; int nmode = 3;
    int ranks[MAXG] = {16, 64, 128, 256}; int nrank = 4;
    double lambdas[MAXG] = {3e-3, 3e-2, 3e-1}; int nlam = 3;
    ds4_loss_weights lw = {1.0f, 0.5f, 0.1f, 1e-3f};
    double dscale = 0.04, trclamp = 0.5; uint64_t seed = 1;
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
        else if (!strcmp(argv[i], "--tr") && i + 1 < argc) trclamp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--fit-ranges") && i + 1 < argc) frs = argv[++i];
        else if (!strcmp(argv[i], "--ev-ranges") && i + 1 < argc) ers = argv[++i];
        else if (!strcmp(argv[i], "--emit-ge") && i + 1 < argc) emitge = argv[++i];
        else if (!strcmp(argv[i], "--emit-z") && i + 1 < argc) { emitz = argv[++i]; g_track_z = 1; }
        else if (!strcmp(argv[i], "--smooth-aug")) g_smooth_aug = 1;
        else if (!strcmp(argv[i], "--dynz")) g_dynz = 1;
        else if (!strcmp(argv[i], "--selftest")) selftest = 1;
        else die("未知参数 %s", argv[i]);
    }
    if (selftest) {                      /* 合成金标: 覆盖网格与行集, 表写 /tmp */
        l0 = l1 = 0; ntok = 2048; nfit = 1536; frs = ers = NULL; out = "/tmp";
        nmode = 2; modes[0] = 1; modes[1] = 2;
        nrank = 1; ranks[0] = 16;
        nlam = 1; lambdas[0] = 3e-3;
    } else if (!anc || !zdir || !out || l0 < 0)
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
        double tl0 = tnow();
        float *X, *R = NULL, *Ys = NULL; int *top1 = NULL;
        zpairs zp = {0};
        if (selftest) st_synth(ntok, &X, &R, &Ys);
        else { X = anchor_load(anc, L, ntok, &top1); zcache_load(zdir, L, ntok, &R, &Ys, &zp); }
        double td = tnow(), tge = td, tx = td, tphi = td, tfta = td;
        float *Yt = xmalloc((size_t)ntok * D * sizeof(float));
        for (size_t i = 0; i < (size_t)ntok * D; i++) Yt[i] = Ys[i] + R[i];
        zl_healthcheck(X, R, Ys, ntok, L);   /* 数据面体检(NaN/rms/巨值集中度) */

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
        fprintf(lf, "# zloss v4 L%02d 裸基线: align=%.6f cls=%.6f total=%.6f\n", L, la0, lc0, tot0);
        fprintf(lf, "# arm M lambda k align classify smooth fixed total ER%% volMB\n");
        printf("★L%d 裸: align %.4f cls %.4f total %.4f\n", L, la0, lc0, tot0);

        float *Yhat = xmalloc((size_t)nev * D * sizeof(float));
        float *Cb = xmalloc((size_t)nev * D * sizeof(float));
        float *Cp = xmalloc((size_t)nev * D * sizeof(float));
        float *zcat = xmalloc((size_t)maxM * maxk * sizeof(float));
        best_t best = { tot0, 0, 0, 0, 0, la0, lc0, NULL };
        float mbest_align[MAXG]; for (int i = 0; i < MAXG; i++) mbest_align[i] = 1e9f;
        int *mode0 = xmalloc((size_t)nev * sizeof(int));
        memset(mode0, 0, (size_t)nev * sizeof(int));
        float *gecorr = NULL, *Rge = NULL, *XP = NULL;
        uint8_t *isfit = NULL;
        float ge_dz[256] = {0}; int ge_wins = 0;   /* GE δ 与"GE 独立赢裸"标记(解耦落地判据) */
        if (emitge && !selftest) {       /* GE-only 端到端注入路: 解算→闸→bf.GE, 跳过全网格 */
            run_emit_ge(emitge, L, &zp, X, R, Ys, Yt_ev, wv, fit, nf, ev, mode0, nev,
                        ntok, &lw, dscale, seed, nth, lf, la0, lc0, tot0, Yhat, Cb, Cp);
            fclose(lf);
            free(X); free(R); free(Ys); free(Yt); free(Yt_ev); free(Ys_ev); free(wv);
            free(Yhat); free(Cb); free(Cp); free(zcat); free(top1); free(mode0);
            free(zp.prow); free(zp.pe); free(zp.pw); free(zp.pyq);
            continue;
        }
        if (!selftest) {  /* 基修正臂组 + 语境门 GEc(段漂移对策: δ_e(c)) */
            run_base_arms(&zp, X, R, Ys, Yt_ev, wv, fit, nf, ev, mode0, nev, ntok,
                          &lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp,
                          &gecorr, &Rge, &isfit, ge_dz, &ge_wins);
            const int cm = 4;
            float *ctx = ctx_build(X, ntok, cm, isfit, L);
            double *thd = xmalloc((size_t)256 * (1 + cm) * sizeof(double));
            solve_gec(&zp, R, ctx, cm, isfit, ntok, 1e-3, L, thd);
            float thf[256 * 16];
            for (int i = 0; i < 256 * (1 + cm); i++) thf[i] = (float)thd[i];
            float *gc = gec_corr_build(&zp, thd, ctx, cm, ntok);
            run_bc_arm("GEc", thf, 256 * (1 + cm), gc, X, Ys, Yt_ev, wv, R, ev, mode0,
                       nev, &lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp);
            free(ctx); free(thd); free(gc);
            if (g_dynz) {        /* ★方案B 动态 z 臂(z=f(x), 四损失判决同表)★ */
                run_dynz_arm(X, R, Ys, Yt_ev, wv, fit, nf, ev, mode0, nev, ntok,
                             ranks, nrank, lambdas, nlam, maxk, &lw, dscale, seed,
                             nth, L, lf, &best, Yhat, Cb, Cp);
                run_pez_arm(&zp, X, R, Ys, Yt_ev, wv, fit, nf, ev, mode0, nev, ntok,
                            lambdas, nlam, maxk, &lw, dscale, seed, nth, L, lf,
                            &best, Yhat, Cb, Cp);
            }
        }
        tge = tnow();

        for (int mi = 0; mi < nmode; mi++) {
            const int M = modes[mi];
            float *C = xmalloc((size_t)M * D * sizeof(float));   /* x 侧门质心 */
            memset(C, 0, (size_t)M * D * sizeof(float));
            int *mode_fit = xmalloc((size_t)nf * sizeof(int));
            int *mode_ev = xmalloc((size_t)nev * sizeof(int));
            memset(mode_fit, 0, (size_t)nf * sizeof(int));
            int table[256]; int gate_route = 0;
            if (M > 1)
                derive_modes(X, R, fit, nf, M, maxk, lambdas[0], seed, L, top1,
                             C, mode_fit, table, &gate_route);
            int cnt[MAXM]; memset(cnt, 0, sizeof cnt);
            for (int i = 0; i < nf; i++) cnt[mode_fit[i]]++;
            for (int i = 0; i < nev; i++)
                mode_ev[i] = (M > 1 && gate_route) ? table[top1[ev[i]]]
                             : mode_assign(X + (size_t)ev[i] * D, C, M);
            printf("  L%d M=%d fit行分布:", L, M);
            for (int m = 0; m < M; m++) printf(" %d", cnt[m]);
            printf("\n");

            ds4_z *(*zsm)[MAXG] = xmalloc(sizeof(ds4_z *[MAXM][MAXG]));
            memset(zsm, 0, sizeof(ds4_z *[MAXM][MAXG]));
            float lamf[MAXG];
            for (int li = 0; li < nlam; li++) lamf[li] = (float)lambdas[li];
            for (int m = 0; m < M; m++) {        /* 每模式一次多 λ 解(Gram 共享) */
                if (!cnt[m]) { printf("  L%d M=%d 模式%d 无fit行, 该模式不修\n", L, M, m); continue; }
                float *Xm = xmalloc((size_t)cnt[m] * D * sizeof(float));
                float *Rm = xmalloc((size_t)cnt[m] * D * sizeof(float));
                int w = 0;
                for (int i = 0; i < nf; i++) if (mode_fit[i] == m) {
                    memcpy(Xm + (size_t)w * D, X + (size_t)fit[i] * D, D * sizeof(float));
                    memcpy(Rm + (size_t)w * D, R + (size_t)fit[i] * D, D * sizeof(float));
                    w++;
                }
                int rk = maxk < cnt[m] ? maxk : cnt[m];
                if (ds4_z_solve_multi(Xm, Rm, (uint32_t)cnt[m], D, D, (uint32_t)rk,
                                      lamf, (uint32_t)nlam, zsm[m]))
                    die("L%d M=%d 模式%d 解算失败", L, M, m);
                free(Xm); free(Rm);
            }
            for (int li = 0; li < nlam; li++) {
                ds4_z *zl[MAXM] = {0};
                for (int m = 0; m < M; m++) zl[m] = zsm[m][li];
                for (int ki = 0; ki < nrank; ki++) {
                    long long vol = (long long)M * D * 4;    /* 质心 f32 也计体积 */
                    int nz = 0;
                    for (int m = 0; m < M; m++) {
                        if (!zl[m]) continue;
                        ds4_z_set_rank(zl[m], (uint32_t)ranks[ki]);
                        vol += (long long)zl[m]->k * (1 + 2 * D) * 2;   /* fp16 z|U|V */
                        for (uint32_t c = 0; c < zl[m]->k; c++) zcat[nz++] = zl[m]->z[c];
                    }
                    if (gate_route) vol += 256;      /* 路由查表门: 每专家 1 字节 */
                    double er = eval_apply(zl, C, M, gate_route, NULL, trclamp, X, Ys, R, ev,
                                           mode_ev, nev, Yhat, Cb, Cp, dscale, seed, nth);
                    float la = ds4_loss_align(Yhat, Yt_ev, (uint32_t)nev, D);
                    float lc = ds4_loss_classify(Yhat, Yt_ev, wv, (uint32_t)nev, D);
                    float ls = ds4_loss_smooth(Cb, Cp, (uint32_t)nev, D);
                    float lfx = ds4_loss_fixed(zcat, (size_t)nz);
                    float tot = ds4_loss_total(&lw, la, lc, ls, lfx);
                    fprintf(lf, "x %d %.3g %d %.6f %.6f %.6f %.6f %.6f %.2f %.2f\n",
                            M, lambdas[li], ranks[ki], la, lc, ls, lfx, tot, er * 100, vol / 1e6);
                    printf("  L%d M=%-2d λ=%-5.3g k=%-3d | align %.4f cls %.4f sm %.5f fx %.5f "
                           "tot %.4f | ER %.1f%% | vol %.1fMB\n",
                           L, M, lambdas[li], ranks[ki], la, lc, ls, lfx, tot, er * 100, vol / 1e6);
                    if (la < mbest_align[mi]) mbest_align[mi] = la;
                    if (tot < best.tot) {
                        best.tot = tot; best.lam = lambdas[li]; best.k = ranks[ki];
                        best.M = M; best.er = er; best.la = la; best.lc = lc; best.arm = "x";
                    }
                    fflush(stdout);
                }
                for (int m = 0; m < M; m++) if (zl[m]) ds4_z_free(zl[m]);
            }
            free(zsm);
            free(C); free(mode_fit); free(mode_ev);
        }
        tx = tnow();
        if (!selftest) {
            /* ---- ftA 臂(冠军特征) / GE+ftA(冠军配方全还原) / ftAw(感知加权解算,
             * cls 方差权进解算目标——四损失进解不只做裁判, L41 定向) ---- */
            XP = build_phi(X, ntok);
            tphi = tnow();
            run_map_arm("ftA", XP, 3 * D, NULL, (float)trclamp, R, R, X, Ys, Yt_ev, wv,
                        NULL, NULL, fit, nf, ev, mode0, nev, ranks, nrank, lambdas, nlam, maxk,
                        &lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp, zcat);
            run_map_arm("GE+ftA", XP, 3 * D, gecorr, (float)trclamp, Rge, R, X, Ys, Yt_ev,
                        wv, NULL, NULL, fit, nf, ev, mode0, nev, ranks, nrank, lambdas, nlam,
                        maxk, &lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp, zcat);
            float *wvf = xmalloc(D * sizeof(float));     /* 权取 fit 侧教师方差, 防 held 泄漏 */
            float *swts = xmalloc(D * sizeof(float));
            {
                float *Ytf = xmalloc((size_t)nf * D * sizeof(float));
                for (int i = 0; i < nf; i++)
                    memcpy(Ytf + (size_t)i * D, Yt + (size_t)fit[i] * D, D * sizeof(float));
                ds4_loss_dim_variance(Ytf, (uint32_t)nf, D, wvf);
                free(Ytf);
                double mw = 0; for (int j = 0; j < D; j++) mw += wvf[j];
                mw = mw / D + 1e-30;
                for (int j = 0; j < D; j++) swts[j] = sqrtf((wvf[j] + (float)(1e-6 * mw)) / (float)mw);
            }
            run_map_arm("ftAw", XP, 3 * D, gecorr, (float)trclamp, Rge, R, X, Ys, Yt_ev,
                        wv, swts, NULL, fit, nf, ev, mode0, nev, ranks, nrank, lambdas, nlam,
                        maxk, &lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp, zcat);
            free(wvf); free(swts);
            for (int ci = 0; ci < nlam; ci++) run_cond_arm(X, R, Ys, Yt_ev, wv, top1,   /* 条件化 z */
                fit, nf, ev, mode0, nev, ntok, &lw, dscale, seed, nth, L, lf, &best,
                Yhat, Cb, Cp, lambdas[ci]);
            run_4l_arm(X, R, Ys, Yt, Yt_ev, wv, fit, nf, ev, mode0, nev, ntok,
                       ranks, nrank, lambdas, nlam, maxk, &lw, dscale, (float)trclamp,
                       seed, nth, L, lf, &best, Yhat, Cb, Cp, zcat, gecorr, Rge);
        }
        tfta = tnow();
        fprintf(stderr, "  [t]L%d 数据%.1f GE%.1f x臂%.1f φ%.1f ftA%.1f 层计%.1fs\n",
                L, td - tl0, tge - td, tx - tge, tphi - tx, tfta - tphi, tnow() - tl0);
        int won = best.arm != NULL;
        fprintf(lf, "# 选中: %s arm=%s M=%d λ=%.3g k=%d total=%.6f (裸 %.6f Δ=%.2f%%) ER=%.1f%%\n",
                won ? "有解" : "全网格输裸(停车审计)", won ? best.arm : "-", best.M,
                best.lam, best.k, best.tot, tot0,
                tot0 > 0 ? 100.0 * (tot0 - best.tot) / tot0 : 0.0, best.er * 100);
        printf("★L%d 终判: %s arm=%s M=%d λ=%.3g k=%d | align %.4f→%.4f cls %.4f→%.4f "
               "total %.4f→%.4f | ER %.1f%%\n",
               L, won ? "选中" : "★全网格输裸=解算器有病, 停车审计★", won ? best.arm : "-",
               best.M, best.lam, best.k, la0, best.la, lc0, best.lc, tot0, best.tot,
               best.er * 100);
        /* ★fclose 必须在 recheck 之后: 原顺序把已 fclose 的 lf 传进 recheck 再
         * fprintf = use-after-free 踩坏 heap(随机层 free(): invalid pointer)★ */
        if (emitz && !selftest) {        /* 部署位宽复评 → 冠军落地(z+GE+四损失参数) */
            z_fp16_recheck(&best, best.zkeep_hasge ? gecorr : NULL, X, Ys, R, Yt_ev,
                           wv, ev, mode0, nev, dscale, seed, nth, L, lf);
            fclose(lf);
            emit_z_finish(emitz, L, &best, ge_dz, la0, lc0,
                          tot0, &lw, Yt, fit, nf, seed, (float)dscale, ge_wins);
        } else fclose(lf);
        if (!won && !emitz) any_lost = 1;   /* 落地跑: 输裸层=记档不注入, 不停车 */
        if (selftest) {
            printf("★selftest: M=1 best align=%.4f  M=2 best align=%.4f\n",
                   mbest_align[0], mbest_align[1]);
            if (!(mbest_align[1] < 0.20f && mbest_align[1] < 0.6f * mbest_align[0]))
                die("selftest 失败: 种进去的两模式结构没被动态 z 收回(布线有病)");
            printf("★SELFTEST PASS(模式布线金标收回)\n");
        }
        fflush(stdout);
        free(X); free(R); free(Ys); free(Yt); free(Yt_ev); free(Ys_ev); free(wv);
        free(Yhat); free(Cb); free(Cp); free(zcat); free(top1);
        free(mode0); free(gecorr); free(Rge); free(XP); free(isfit);
        free(zp.prow); free(zp.pe); free(zp.pw); free(zp.pyq);
    }
    free(fit); free(ev);
    if (any_lost) { fprintf(stderr, "zloss_solve: 有层全网格输裸 — 停车审计(exit 3)\n"); return 3; }
    return 0;
}
