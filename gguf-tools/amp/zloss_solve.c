/* zloss_solve.c — 单层贪心最优反修解算器(2026-08-26 用户设计定案)。
 *
 * 设计(用户四点, 逐条落地):
 *   ① 单层贪心最优, 不管链态: 每层独立解, 判据=本层 held 四损失最小, 不建模层间耦合。
 *   ② z 与当层全部量化计算一起算: 学生 = 引擎捕获真值(x̂=ffn_in, 输出=ffn_out,
 *      attn/shared/router/norm/专家的量化效应全在里面), 不是 Python 理想化重算;
 *      教师 = FP 锚口径 routed(teacher_routed --anchor 产)。R = 教师 − 学生。
 *   ③ 四损失 = ds4_loss 模块(ALGORITHM.md §4): L_align + L_classify(方差感知权)
 *      + L_smooth(固定种子 dither) + L_fixed(z 能量), (λ,k) 网格 held 择优;
 *      k=0(不修)也参赛 —— 贪心的诚实基线, 输给裸就判空。
 *   ④ 反修与引擎复用: 解算 ds4_z_solve / 评估与运行时 ds4_z_apply 是同一份模块
 *      代码(#include 仓库根 ds4_z.c, 与引擎 MODULE_OBJS 同源)。
 *
 * 产物(都是有体积的文件):
 *   --out/z_L%02d.ds4z        ds4_z_save 序列化(f32 U|V|z, 引擎侧同格式加载)
 *   --out/fourloss_L%02d.txt  四损失记录: 择优网格全表 + 选中 (λ,k) + 字节账
 *   --dql 给了则注入 zl.RRR(fp16 z|U|V, tr=0.5) 进 dql_L%02d.bin 供冻结判决尺
 *   (ds4quant_run.old) 回放; zloss_manifest.txt 账本记原长, 重跑先截回(幂等)。
 *
 * 用法: zloss_solve --cap DIR --out DIR --layers a-b [--dql DIR]
 *   [--ntok 8192] [--nfit 6144] [--ranks 16,64] [--lambdas 3e-3,3e-2]
 *   [--wa 1] [--wc 0.5] [--ws 0.1] [--wf 1e-3] [--dither 0.04] [--seed 1]
 *   [--fit-ranges a:b,..] [--ev-ranges a:b,..]   (行掩码: 拼接语料剔污染行)
 * 全 CLI 参数, 无环境变量(铁律)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include "npy.h"
#include "ds4_z.c"                      /* -I.. 仓库根: 与引擎同一份实现(复用铁律) */
#include "ds4_loss.c"
#include "src/common/ds4_float.h"       /* ds4_f64_to_f16: 注入载荷用共享转换 */

#define D 4096
#define MAXG 8                          /* λ/k 网格上限 */

static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "zloss_solve Error: "); vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n"); va_end(ap); exit(1);
}
static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) die("OOM %zu", n); return p; }

/* "a,b,c" → 数组 */
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
/* "a:b,c:d" → 行号数组(左闭右开) */
static int *parse_ranges(const char *s, int *n_out) {
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

static float *load_rows(const char *dir, const char *name, int L, int ntok) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s_L%d.npy", dir, name, L);
    npy_meta m; float *a = npy_read_f32(p, &m);
    if (!a) die("%s 读不出", p);
    if (m.ndim != 2 || m.shape[1] != D || m.shape[0] < ntok)
        die("%s 形状不对 [%lld×%lld], 需 ≥%d×%d", p, (long long)m.shape[0], (long long)m.shape[1], ntok, D);
    return a;                            /* 只用前 ntok 行 */
}

/* ---- ev 行评估(pthread 并行): Yhat=Ys+M(x), Cbase=M(x), Cpert=M(x+δ) ---- */
typedef struct {
    const ds4_z *zl; const float *X, *Ys; const int *ev; int nev;
    float *Yhat, *Cb, *Cp; double dscale; uint64_t seed;
    int t0, t1;
} ev_ctx;
static void *ev_worker(void *arg) {
    ev_ctx *c = (ev_ctx *)arg;
    float *delta = xmalloc(D * sizeof(float));
    float *xp = xmalloc(D * sizeof(float));
    for (int i = c->t0; i < c->t1; i++) {
        const float *x = c->X + (size_t)c->ev[i] * D;
        float *yh = c->Yhat + (size_t)i * D, *cb = c->Cb + (size_t)i * D, *cp = c->Cp + (size_t)i * D;
        memcpy(yh, c->Ys + (size_t)c->ev[i] * D, D * sizeof(float));
        memset(cb, 0, D * sizeof(float)); memset(cp, 0, D * sizeof(float));
        ds4_z_apply(c->zl, x, yh);       /* 引擎同一份 apply */
        ds4_z_apply(c->zl, x, cb);
        double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
        const float rms = (float)sqrt(ss / D);
        ds4_loss_dither(c->seed, (uint32_t)c->ev[i], (float)(c->dscale * rms), delta, D);
        for (int j = 0; j < D; j++) xp[j] = x[j] + delta[j];
        ds4_z_apply(c->zl, xp, cp);
    }
    free(delta); free(xp); return NULL;
}
static void eval_apply(const ds4_z *zl, const float *X, const float *Ys, const int *ev, int nev,
                       float *Yhat, float *Cb, float *Cp, double dscale, uint64_t seed, int nth) {
    pthread_t th[32]; ev_ctx cx[32];
    if (nth > 32) nth = 32;
    for (int t = 0; t < nth; t++) {
        cx[t] = (ev_ctx){zl, X, Ys, ev, nev, Yhat, Cb, Cp, dscale, seed,
                         nev * t / nth, nev * (t + 1) / nth};
        pthread_create(&th[t], NULL, ev_worker, &cx[t]);
    }
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);
}

int main(int argc, char **argv) {
    const char *cap = NULL, *out = NULL, *dql = NULL, *frs = NULL, *ers = NULL;
    int l0 = -1, l1 = -1, ntok = 8192, nfit = 6144, nth = 16;
    int ranks[MAXG] = {16, 64}; int nrank = 2;
    double lambdas[MAXG] = {3e-3, 3e-2}; int nlam = 2;
    ds4_loss_weights lw = {1.0f, 0.5f, 0.1f, 1e-3f};
    double dscale = 0.04; uint64_t seed = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--dql") && i + 1 < argc) dql = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d-%d", &l0, &l1) != 2) die("--layers a-b");
        }
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nfit") && i + 1 < argc) nfit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nth = atoi(argv[++i]);
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
    if (!cap || !out || l0 < 0)
        die("用法: zloss_solve --cap DIR --out DIR --layers a-b [--dql DIR] ...(见文件头)");

    /* fit/ev 行(默认前 nfit 拟合、其余 held; 行掩码给了则以掩码为准) */
    int nf, nev, *fit, *ev;
    if (frs) { if (!ers) die("--fit-ranges 设了必须同时给 --ev-ranges"); }
    if (frs) { fit = parse_ranges(frs, &nf); ev = parse_ranges(ers, &nev); }
    else {
        nf = nfit; nev = ntok - nfit;
        fit = xmalloc((size_t)nf * sizeof(int)); ev = xmalloc((size_t)nev * sizeof(int));
        for (int i = 0; i < nf; i++) fit[i] = i;
        for (int i = 0; i < nev; i++) ev[i] = nfit + i;
    }
    for (int i = 0; i < nf; i++) if (fit[i] >= ntok) die("fit 行 %d ≥ ntok", fit[i]);
    for (int i = 0; i < nev; i++) if (ev[i] >= ntok) die("ev 行 %d ≥ ntok", ev[i]);
    int maxk = 0; for (int i = 0; i < nrank; i++) if (ranks[i] > maxk) maxk = ranks[i];
    printf("zloss_solve L%d-%d  fit=%d ev=%d  ranks=%d 档(max %d)  λ=%d 档  "
           "w=[%.3g %.3g %.3g %.3g]  dither=%.3g seed=%llu\n",
           l0, l1, nf, nev, nrank, maxk, nlam,
           lw.w_align, lw.w_classify, lw.w_smooth, lw.w_fixed, dscale,
           (unsigned long long)seed);

    long long vol_z = 0, vol_loss = 0;
    for (int L = l0; L <= l1; L++) {
        float *X = load_rows(cap, "ffn_in", L, ntok);
        float *Ys = load_rows(cap, "obase_v3", L, ntok);
        float *Yt = load_rows(cap, "routed", L, ntok);
        float *R = xmalloc((size_t)ntok * D * sizeof(float));
        for (size_t i = 0; i < (size_t)ntok * D; i++) R[i] = Yt[i] - Ys[i];

        /* fit 行紧凑打包 */
        float *Xf = xmalloc((size_t)nf * D * sizeof(float));
        float *Rf = xmalloc((size_t)nf * D * sizeof(float));
        for (int i = 0; i < nf; i++) {
            memcpy(Xf + (size_t)i * D, X + (size_t)fit[i] * D, D * sizeof(float));
            memcpy(Rf + (size_t)i * D, R + (size_t)fit[i] * D, D * sizeof(float));
        }
        /* held 教师与感知列权(ds4_loss 模块: 按 y 的 per-dim 方差) */
        float *Yt_ev = xmalloc((size_t)nev * D * sizeof(float));
        float *Ys_ev = xmalloc((size_t)nev * D * sizeof(float));
        for (int i = 0; i < nev; i++) {
            memcpy(Yt_ev + (size_t)i * D, Yt + (size_t)ev[i] * D, D * sizeof(float));
            memcpy(Ys_ev + (size_t)i * D, Ys + (size_t)ev[i] * D, D * sizeof(float));
        }
        float *wv = xmalloc(D * sizeof(float));
        ds4_loss_dim_variance(Yt_ev, (uint32_t)nev, D, wv);

        /* 裸基线(k=0): 输给它的解一律判空 —— 单层贪心的诚实底线 */
        float la0 = ds4_loss_align(Ys_ev, Yt_ev, (uint32_t)nev, D);
        float lc0 = ds4_loss_classify(Ys_ev, Yt_ev, wv, (uint32_t)nev, D);
        float tot0 = ds4_loss_total(&lw, la0, lc0, 0.0f, 0.0f);

        float *Yhat = xmalloc((size_t)nev * D * sizeof(float));
        float *Cb = xmalloc((size_t)nev * D * sizeof(float));
        float *Cp = xmalloc((size_t)nev * D * sizeof(float));
        char lossp[1024]; snprintf(lossp, sizeof lossp, "%s/fourloss_L%02d.txt", out, L);
        FILE *lf = fopen(lossp, "w"); if (!lf) die("%s 写不开", lossp);
        fprintf(lf, "# zloss L%02d  裸基线: align=%.6f cls=%.6f total=%.6f\n", L, la0, lc0, tot0);
        fprintf(lf, "# lambda k align classify smooth fixed total\n");

        ds4_z *best = NULL; double best_tot = tot0, best_lam = 0; int best_k = 0;
        float bla = la0, blc = lc0, bls = 0, blf = 0;
        for (int li = 0; li < nlam; li++) {
            ds4_z *zl = ds4_z_solve(Xf, Rf, (uint32_t)nf, D, D, (uint32_t)maxk, (float)lambdas[li]);
            if (!zl) die("L%d λ=%g 解算失败(非正定?)", L, lambdas[li]);
            for (int ki = 0; ki < nrank; ki++) {
                ds4_z_set_rank(zl, (uint32_t)ranks[ki]);
                eval_apply(zl, X, Ys, ev, nev, Yhat, Cb, Cp, dscale, seed, nth);
                float la = ds4_loss_align(Yhat, Yt_ev, (uint32_t)nev, D);
                float lc = ds4_loss_classify(Yhat, Yt_ev, wv, (uint32_t)nev, D);
                float ls = ds4_loss_smooth(Cb, Cp, (uint32_t)nev, D);
                float lfx = ds4_loss_fixed(zl->z, zl->k);
                float tot = ds4_loss_total(&lw, la, lc, ls, lfx);
                fprintf(lf, "%.3g %d %.6f %.6f %.6f %.6f %.6f\n",
                        lambdas[li], ranks[ki], la, lc, ls, lfx, tot);
                if (tot < best_tot) {
                    best_tot = tot; best_lam = lambdas[li]; best_k = ranks[ki];
                    bla = la; blc = lc; bls = ls; blf = lfx;
                    if (best && best != zl) ds4_z_free(best);
                    best = zl;
                }
            }
            if (best != zl) ds4_z_free(zl);
        }

        long long zbytes = 0;
        if (best) {
            ds4_z_set_rank(best, (uint32_t)best_k);
            char zp[1024]; snprintf(zp, sizeof zp, "%s/z_L%02d.ds4z", out, L);
            if (ds4_z_save(best, zp) != 0) die("L%d z 保存失败", L);
            FILE *zf = fopen(zp, "rb"); fseek(zf, 0, SEEK_END); zbytes = ftell(zf); fclose(zf);
            vol_z += zbytes;
        }
        fprintf(lf, "# 选中: %s  λ=%.3g k=%d total=%.6f (裸 %.6f, Δ=%.2f%%)  z字节=%lld\n",
                best ? "有解" : "判空(裸更优)", best_lam, best_k, best_tot, tot0,
                tot0 > 0 ? 100.0 * (tot0 - best_tot) / tot0 : 0.0, zbytes);
        fclose(lf);
        { FILE *sf = fopen(lossp, "rb"); fseek(sf, 0, SEEK_END); vol_loss += ftell(sf); fclose(sf); }

        /* 判空层重跑安全: 上一轮若注入过, 必须截回原长 —— 否则陈旧记录冒充本轮产物 */
        if (!best && dql) {
            char dp[1024], mp[1024];
            snprintf(dp, sizeof dp, "%s/dql_L%02d.bin", dql, L);
            snprintf(mp, sizeof mp, "%s/zloss_manifest.txt", dql);
            FILE *mf = fopen(mp, "r");
            long long orig = -1;
            if (mf) { char ln[128]; int ml; long long mo;
                while (fgets(ln, sizeof ln, mf))
                    if (sscanf(ln, "L=%d orig=%lld", &ml, &mo) == 2 && ml == L) orig = mo;
                fclose(mf); }
            if (orig >= 0) {
                FILE *df = fopen(dp, "r+b");
                if (df) { if (ftruncate(fileno(df), (off_t)orig) != 0) die("L%d 判空截回失败", L);
                          fclose(df); }
            }
        }
        /* 注入 zl.RRR 供冻结判决尺(格式=zlayer 同款: 116B 记录头 + u32 k|f32 tr|u32 din|u32 dout|fp16 z|U|V) */
        if (best && dql) {
            char dp[1024], mp[1024];
            snprintf(dp, sizeof dp, "%s/dql_L%02d.bin", dql, L);
            snprintf(mp, sizeof mp, "%s/zloss_manifest.txt", dql);
            long long orig = -1;
            FILE *mf = fopen(mp, "r");
            if (mf) { char ln[128]; int ml; long long mo;
                while (fgets(ln, sizeof ln, mf))
                    if (sscanf(ln, "L=%d orig=%lld", &ml, &mo) == 2 && ml == L) orig = mo;
                fclose(mf); }
            FILE *df = fopen(dp, "r+b"); if (!df) die("%s 打不开", dp);
            fseek(df, 0, SEEK_END); long long cur = ftell(df);
            if (orig < 0) { orig = cur; mf = fopen(mp, "a"); fprintf(mf, "L=%d orig=%lld\n", L, orig); fclose(mf); }
            else if (ftruncate(fileno(df), (off_t)orig) != 0) die("L%d 截回原长失败", L);
            fseek(df, orig, SEEK_SET);
            const int k = best_k;
            size_t nh = (size_t)k + (size_t)D * k + (size_t)D * k;
            size_t psz = 16 + nh * 2;
            uint8_t *rec = xmalloc(116 + psz);
            memset(rec, 0, 116 + psz);
            memcpy(rec, "zl.RRR", 6);
            uint64_t p64 = psz; memcpy(rec + 88, &p64, 8);
            int32_t one = 1; memcpy(rec + 112, &one, 4);
            uint8_t *pay = rec + 116;
            uint32_t u32k = (uint32_t)k, u32di = D, u32do = D; float tr05 = 0.5f;
            memcpy(pay, &u32k, 4); memcpy(pay + 4, &tr05, 4);
            memcpy(pay + 8, &u32di, 4); memcpy(pay + 12, &u32do, 4);
            uint16_t *h = (uint16_t *)(pay + 16), *U16 = h + k, *V16 = h + k + (size_t)D * k;
            for (int c = 0; c < k; c++) h[c] = ds4_f64_to_f16((double)best->z[c]);
            for (int j = 0; j < D; j++) for (int c = 0; c < k; c++)
                U16[(size_t)j * k + c] = ds4_f64_to_f16((double)best->U[(size_t)j * best->rank + c]);
            for (int i = 0; i < D; i++) for (int c = 0; c < k; c++)
                V16[(size_t)i * k + c] = ds4_f64_to_f16((double)best->V[(size_t)i * best->rank + c]);
            if (fwrite(rec, 1, 116 + psz, df) != 116 + psz) die("L%d 注入写失败", L);
            fclose(df); free(rec);
        }
        printf("★L%d 四损失: align %.4f→%.4f  cls %.4f→%.4f  smooth %.5f  fixed %.5f | "
               "total %.4f→%.4f (%s λ=%.3g k=%d) | z %.1fMB%s\n",
               L, la0, bla, lc0, blc, bls, blf, tot0, best_tot,
               best ? "选中" : "判空", best_lam, best_k, zbytes / 1e6,
               (best && dql) ? " 注入✓" : "");
        fflush(stdout);
        if (best) ds4_z_free(best);
        free(X); free(Ys); free(Yt); free(R); free(Xf); free(Rf);
        free(Yt_ev); free(Ys_ev); free(wv); free(Yhat); free(Cb); free(Cp);
    }
    printf("zloss_solve 收官 L%d-%d  体积: z=%.1fMB fourloss=%.1fKB\n",
           l0, l1, vol_z / 1e6, vol_loss / 1e3);
    free(fit); free(ev);
    return 0;
}
