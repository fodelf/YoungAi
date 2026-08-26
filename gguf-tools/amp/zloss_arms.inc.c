/* zloss_arms.inc.c — GE 臂与 ftA 臂(物理分片, 只被 zloss_solve.c include)。
 *
 * 依据(L20 针终判): 裸 x 全家(静态+EM模式)在 FP 自洽口径下输裸, 而 r64c 冠军
 * 同层同口径有正收益 —— 配方差恰是两样: ①GE(bf.GE 每专家增益, 门=路由本身,
 * 部署免费不用学) ②ftA φ=[x, x⊙x/rms, relu(x)] 非线性特征提升。两臂全是复用:
 * GE 与 zlayer bf.GE 同族(四损失口径重解, 求解走 linalg_small chol_solve_spd),
 * φ 与引擎 type6 ftA 加载路同式(公式必须逐字对齐, 否则解算目标≠部署行为)。 */

/* eval_apply 在主文件后段定义(worker 按 zl->d_in 自动 φ 提升), 此处前置声明 */
static double eval_apply(ds4_z *const *zl, const float *C, int M, int gate_route,
                         const float *bc, float tr, const float *X, const float *Ys,
                         const float *R, const int *ev, const int *mode_ev, int nev,
                         float *Yhat, float *Cb, float *Cp, double dscale,
                         uint64_t seed, int nth);

/* φ(x) = [x, x⊙x/rms, relu(x)], rms=sqrt(mean x²)+1e-6 — zlayer/rec_fidelity 同式 */
static void mk_phi(const float *x, float *p) {
    double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
    float nr = (float)sqrt(ss / D) + 1e-6f;
    for (int j = 0; j < D; j++) {
        p[j] = x[j];
        p[D + j] = x[j] * x[j] / nr;
        p[2 * D + j] = x[j] > 0 ? x[j] : 0;
    }
}
static float *build_phi(const float *X, int n) {
    float *P = xmalloc((size_t)n * 3 * D * sizeof(float));
    for (int i = 0; i < n; i++) mk_phi(X + (size_t)i * D, P + (size_t)i * 3 * D);
    return P;
}

/* zcache 配对(路由精确条件化的数据面): prow/pe/pw/pyq */
typedef struct { int *prow, *pe; float *pw, *pyq; long long npair; } zpairs;

/* GE 闭式解: min Σ_t∈fit ||R_t − Σ_{e∈S_t} δ_e·w·pYQ||² + λ·量纲化ridge。
 * A 256×256 正规方程, chol_solve_spd(linalg_small, 全仓唯一小 Cholesky)。 */
static void solve_ge(const zpairs *zp, const float *R, const uint8_t *isfit,
                     int ntok, double lam, int L, double *delta) {
    long long *rs = xmalloc((size_t)(ntok + 1) * sizeof(long long));
    memset(rs, 0, (size_t)(ntok + 1) * sizeof(long long));
    for (long long i = 0; i < zp->npair; i++) if (zp->prow[i] < ntok) rs[zp->prow[i] + 1]++;
    for (int t = 0; t < ntok; t++) rs[t + 1] += rs[t];
    long long *ord = xmalloc((size_t)zp->npair * sizeof(long long));
    { long long *cur = xmalloc((size_t)ntok * sizeof(long long));
      memcpy(cur, rs, (size_t)ntok * sizeof(long long));
      for (long long i = 0; i < zp->npair; i++)
          if (zp->prow[i] < ntok) ord[cur[zp->prow[i]]++] = i;
      free(cur); }
    double *A = xmalloc((size_t)256 * 256 * sizeof(double));
    memset(A, 0, (size_t)256 * 256 * sizeof(double));
    memset(delta, 0, 256 * sizeof(double));
    for (int t = 0; t < ntok; t++) {
        if (!isfit[t]) continue;
        const float *r = R + (size_t)t * D;
        for (long long a = rs[t]; a < rs[t + 1]; a++) {
            long long i = ord[a];
            const float *yi = zp->pyq + (size_t)i * D;
            double wi = zp->pw[i]; int e = zp->pe[i];
            double bi = 0;
            for (int j = 0; j < D; j++) bi += (double)yi[j] * r[j];
            delta[e] += wi * bi;
            for (long long b2 = a; b2 < rs[t + 1]; b2++) {
                long long k = ord[b2];
                const float *yk = zp->pyq + (size_t)k * D;
                double d = 0;
                for (int j = 0; j < D; j++) d += (double)yi[j] * yk[j];
                d *= wi * zp->pw[k];
                int f = zp->pe[k];
                A[(size_t)e * 256 + f] += d;
                if (k != i) A[(size_t)f * 256 + e] += d;
            }
        }
    }
    double md = 0; for (int e = 0; e < 256; e++) md += A[(size_t)e * 256 + e];
    md = md / 256 + 1e-30;
    for (int e = 0; e < 256; e++) A[(size_t)e * 256 + e] += lam * md;
    if (chol_solve_spd(A, 256, delta, 1) != 0) {
        for (int e = 0; e < 256; e++) A[(size_t)e * 256 + e] += 100 * lam * md;
        if (chol_solve_spd(A, 256, delta, 1) != 0) die("L%d GE 正规方程非正定", L);
    }
    free(rs); free(ord); free(A);
}

/* corr[t] = Σ_{e∈S_t} δ_e·w·pYQ — 全行一次построить, fit/ev 共用 */
static float *ge_corr_build(const zpairs *zp, const double *delta, int ntok) {
    float *corr = xmalloc((size_t)ntok * D * sizeof(float));
    memset(corr, 0, (size_t)ntok * D * sizeof(float));
    for (long long i = 0; i < zp->npair; i++) {
        int t = zp->prow[i]; if (t >= ntok) continue;
        double d = delta[zp->pe[i]] * zp->pw[i];
        if (fabs(d) < 1e-12) continue;
        float *c = corr + (size_t)t * D;
        const float *y = zp->pyq + (size_t)i * D;
        for (int j = 0; j < D; j++) c[j] += (float)(d * y[j]);
    }
    return corr;
}

/* 择优追踪(全臂共享) */
typedef struct {
    double tot, lam, er; int M, k; float la, lc; const char *arm;
} best_t;

/* 线性/ftA 图臂: 特征 XF(din 宽), 靶 Reff(bc 臂=R−GE修正), M=1 静态。
 * 评估仍走 eval_apply(worker 按 zl->d_in 自动 φ 提升), ER 恒对原始 R。 */
static void run_map_arm(const char *arm, const float *XF, int din, const float *bc,
                        float tr, const float *Reff, const float *R, const float *X,
                        const float *Ys, const float *Yt_ev, const float *wv,
                        const int *fit, int nf, const int *ev, const int *mode0, int nev,
                        const int *ranks, int nrank, const double *lambdas, int nlam,
                        int maxk, const ds4_loss_weights *lw, double dscale,
                        uint64_t seed, int nth, int L, FILE *lf, best_t *best,
                        float *Yhat, float *Cb, float *Cp, float *zcat) {
    float *Xm = xmalloc((size_t)nf * din * sizeof(float));
    float *Rm = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        memcpy(Xm + (size_t)i * din, XF + (size_t)fit[i] * din, (size_t)din * sizeof(float));
        memcpy(Rm + (size_t)i * D, Reff + (size_t)fit[i] * D, D * sizeof(float));
    }
    for (int li = 0; li < nlam; li++) {
        ds4_z *zl[1];
        zl[0] = ds4_z_solve(Xm, Rm, (uint32_t)nf, (uint32_t)din, D, (uint32_t)maxk,
                            (float)lambdas[li]);
        if (!zl[0]) die("L%d %s λ=%g 解算失败", L, arm, lambdas[li]);
        for (int ki = 0; ki < nrank; ki++) {
            ds4_z_set_rank(zl[0], (uint32_t)ranks[ki]);
            long long vol = (long long)zl[0]->k * (1 + din + D) * 2 + (bc ? 512 : 0);
            int nz = 0;
            for (uint32_t c = 0; c < zl[0]->k; c++) zcat[nz++] = zl[0]->z[c];
            double er = eval_apply(zl, NULL, 1, 0, bc, tr, X, Ys, R, ev, mode0, nev,
                                   Yhat, Cb, Cp, dscale, seed, nth);
            float la = ds4_loss_align(Yhat, Yt_ev, (uint32_t)nev, D);
            float lc = ds4_loss_classify(Yhat, Yt_ev, wv, (uint32_t)nev, D);
            float ls = ds4_loss_smooth(Cb, Cp, (uint32_t)nev, D);
            float lfx = ds4_loss_fixed(zcat, (size_t)nz);
            float tot = ds4_loss_total(lw, la, lc, ls, lfx);
            fprintf(lf, "%s 1 %.3g %d %.6f %.6f %.6f %.6f %.6f %.2f %.2f\n",
                    arm, lambdas[li], ranks[ki], la, lc, ls, lfx, tot, er * 100, vol / 1e6);
            printf("  L%d %-6s λ=%-5.3g k=%-3d | align %.4f cls %.4f sm %.5f fx %.5f "
                   "tot %.4f | ER %.1f%% | vol %.1fMB\n",
                   L, arm, lambdas[li], ranks[ki], la, lc, ls, lfx, tot, er * 100, vol / 1e6);
            if (tot < best->tot) {
                best->tot = tot; best->lam = lambdas[li]; best->k = ranks[ki];
                best->M = 1; best->er = er; best->la = la; best->lc = lc; best->arm = arm;
            }
            fflush(stdout);
        }
        ds4_z_free(zl[0]);
    }
    free(Xm); free(Rm);
}

/* GE 单臂评估: 图=空, bc=GE 修正 */
static void run_ge_arm(const double *delta, const float *gecorr, const float *X,
                       const float *Ys, const float *Yt_ev, const float *wv,
                       const float *R, const int *ev, const int *mode0, int nev,
                       const ds4_loss_weights *lw, double dscale, uint64_t seed,
                       int nth, int L, FILE *lf, best_t *best,
                       float *Yhat, float *Cb, float *Cp) {
    ds4_z *zl[1] = {0};
    double er = eval_apply(zl, NULL, 1, 1 /*路由门语义: 扰动不改 GE*/, gecorr, 0,
                           X, Ys, R, ev, mode0, nev, Yhat, Cb, Cp, dscale, seed, nth);
    float dz[256];
    for (int e = 0; e < 256; e++) dz[e] = (float)delta[e];
    float la = ds4_loss_align(Yhat, Yt_ev, (uint32_t)nev, D);
    float lc = ds4_loss_classify(Yhat, Yt_ev, wv, (uint32_t)nev, D);
    float ls = ds4_loss_smooth(Cb, Cp, (uint32_t)nev, D);
    float lfx = ds4_loss_fixed(dz, 256);
    float tot = ds4_loss_total(lw, la, lc, ls, lfx);
    int nact = 0; for (int e = 0; e < 256; e++) if (fabs(delta[e]) > 1e-4) nact++;
    fprintf(lf, "GE 1 0 0 %.6f %.6f %.6f %.6f %.6f %.2f 0.00\n", la, lc, ls, lfx, tot, er * 100);
    printf("  L%d GE     活门=%d      | align %.4f cls %.4f sm %.5f fx %.5f "
           "tot %.4f | ER %.1f%% | vol 0.0MB\n", L, nact, la, lc, ls, lfx, tot, er * 100);
    if (tot < best->tot) {
        best->tot = tot; best->lam = 0; best->k = 0; best->M = 1;
        best->er = er; best->la = la; best->lc = lc; best->arm = "GE";
    }
    fflush(stdout);
}
