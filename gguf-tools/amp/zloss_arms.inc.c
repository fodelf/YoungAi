/* zloss_arms.inc.c — zloss_solve 的解算臂分片(500 行守卫所迫的唯一伴生文件,
 * 语义化命名铁律 2026-08-26): 合成金标 + GE/ftA 臂 + EM 模式发现 + zcache 数据面。
 * 解算器仍是一份代码(zloss_solve.c 原地迭代), 本片只是它放不下的臂实现。 */
static int mode_assign(const float *x, const float *C, int M);   /* 主文件后段定义 */

/* zloss_selftest.inc.c — zloss_solve 的合成金标(物理分片, 只被 zloss_solve.c
 * include)。种两模式各一张 rank-8 线性图, M=2 必须近零收回, M=1 必须收不动
 * —— 验的是 残差聚类→x 门→模式打包→eval 的全布线, 不是理论。
 *
 * 金标形态(两次翻案的教训都固化在此):
 * ①x = B·h 住 H=64 维流形 —— 真激活是低维各向异性的, 金标必须同构; 满维各向
 *   同性随机 x 下 1.5k 行撑不起 4096² 图的泛化, 谁来解都收不回(数据墙非布线针)。
 * ②模式种在"靶怎么依赖 x"里(sign(p·h) 选 U_c·G_c), x 的密度分布对两模式完全
 *   对称 —— x 侧无监督聚类原理上看不见这刀怎么切(任意对径切分密度等价),
 *   逼着解算器走 残差方向聚类+x 侧门 的路(gate(x) 支柱)。 */
static uint64_t st_s = 0x9E3779B97F4A7C15ULL;
static float st_u(void) {
    st_s = st_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int64_t)(st_s >> 33) % 2000001 - 1000000) / 1e6f;
}
static void st_synth(int ntok, float **Xo, float **Ro, float **Yso) {
    const int RK = 8, H = 64;
    float *X = xmalloc((size_t)ntok * D * 4), *R = xmalloc((size_t)ntok * D * 4);
    float *Ys = xmalloc((size_t)ntok * D * 4);
    float *B = xmalloc((size_t)D * H * 4), *ph = xmalloc(H * 4), *h = xmalloc(H * 4);
    float *U = xmalloc((size_t)2 * RK * D * 4), *G = xmalloc((size_t)2 * RK * H * 4);
    for (size_t i = 0; i < (size_t)D * H; i++) B[i] = st_u();
    for (int t = 0; t < H; t++) ph[t] = st_u();
    for (size_t i = 0; i < (size_t)2 * RK * D; i++) U[i] = st_u();
    for (size_t i = 0; i < (size_t)2 * RK * H; i++) G[i] = st_u();
    memset(Ys, 0, (size_t)ntok * D * 4);
    for (int i = 0; i < ntok; i++) {
        float *x = X + (size_t)i * D, *r = R + (size_t)i * D;
        double dp = 0;
        for (int t = 0; t < H; t++) { h[t] = st_u(); dp += (double)h[t] * ph[t]; }
        int c = dp > 0;
        for (int j = 0; j < D; j++) {
            double a = 0; const float *b = B + (size_t)j * H;
            for (int t = 0; t < H; t++) a += (double)b[t] * h[t];
            x[j] = (float)a;
        }
        memset(r, 0, D * 4);
        for (int rr = 0; rr < RK; rr++) {
            const float *g = G + ((size_t)c * RK + rr) * H, *u = U + ((size_t)c * RK + rr) * D;
            double a = 0; for (int t = 0; t < H; t++) a += (double)g[t] * h[t];
            a /= H;
            for (int j = 0; j < D; j++) r[j] += (float)(a * u[j]);
        }
    }
    free(B); free(ph); free(h); free(U); free(G);
    *Xo = X; *Ro = R; *Yso = Ys;
}

/* zloss_arms.inc.c — GE 臂与 ftA 臂(物理分片, 只被 zloss_solve.c include)。
 *
 * 依据(L20 针终判): 裸 x 全家(静态+EM模式)在 FP 自洽口径下输裸, 而 r64c 冠军
 * 同层同口径有正收益 —— 配方差恰是两样: ①GE(bf.GE 每专家增益, 门=路由本身,
 * 部署免费不用学) ②ftA φ=[x, x⊙x/rms, relu(x)] 非线性特征提升。两臂全是复用:
 * GE 与 zlayer bf.GE 同族(四损失口径重解, 求解走 linalg_small chol_solve_spd),
 * φ 与引擎 type6 ftA 加载路同式(公式必须逐字对齐, 否则解算目标≠部署行为)。 */

/* ---- CLI 解析工具(主文件 500 行腾挪迁入, 逐字不动) ---- */
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

/* 冠军 z 截秩克隆(--emit-z 序列化用): 只留活跃 k 列, U/V 步长 rank→k */
static ds4_z *z_clone_k(const ds4_z *s) {
    ds4_z *c = xmalloc(sizeof(ds4_z));
    c->d_in = s->d_in; c->d_out = s->d_out; c->rank = s->k; c->k = s->k;
    c->z = xmalloc((size_t)s->k * sizeof(float));
    c->U = xmalloc((size_t)s->d_out * s->k * sizeof(float));
    c->V = xmalloc((size_t)s->d_in * s->k * sizeof(float));
    memcpy(c->z, s->z, (size_t)s->k * sizeof(float));
    for (uint32_t j = 0; j < s->d_out; j++)
        memcpy(c->U + (size_t)j * s->k, s->U + (size_t)j * s->rank, (size_t)s->k * sizeof(float));
    for (uint32_t i = 0; i < s->d_in; i++)
        memcpy(c->V + (size_t)i * s->k, s->V + (size_t)i * s->rank, (size_t)s->k * sizeof(float));
    return c;
}

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

/* GE 闭式解: min Σ_t∈fit ||diag(sw)(R_t − Σ_{e∈S_t} δ_e·w·pYQ)||² + λ·量纲化ridge。
 * sw2=每通道权²(cls 方差权白化=四损失 L_classify 进解算目标, 用户令), NULL=平权。
 * A 256×256 正规方程, chol_solve_spd(linalg_small, 全仓唯一小 Cholesky)。 */
static void solve_ge(const zpairs *zp, const float *R, const uint8_t *isfit,
                     int ntok, double lam, int L, double *delta, const float *sw2) {
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
            if (sw2) for (int j = 0; j < D; j++) bi += (double)yi[j] * r[j] * sw2[j];
            else for (int j = 0; j < D; j++) bi += (double)yi[j] * r[j];
            delta[e] += wi * bi;
            for (long long b2 = a; b2 < rs[t + 1]; b2++) {
                long long k = ord[b2];
                const float *yk = zp->pyq + (size_t)k * D;
                double d = 0;
                if (sw2) for (int j = 0; j < D; j++) d += (double)yi[j] * yk[j] * sw2[j];
                else for (int j = 0; j < D; j++) d += (double)yi[j] * yk[j];
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

/* 择优追踪(全臂共享)。zkeep=冠军 z 克隆(--emit-z 时跟踪); zkeep_hasge=z 是 GE 基上
 * 的叠加(序列化须连 bf.GE); bcdz/bckind=冠军基修正臂参数(1=256专家门可序列化, 2=无路)。 */
typedef struct {
    double tot, lam, er; int M, k; float la, lc; const char *arm;
    ds4_z *zkeep; int zkeep_hasge;
    float bcdz[256]; int bckind;
} best_t;

/* 线性/ftA 图臂: 特征 XF(din 宽), 靶 Reff(bc 臂=R−GE修正), M=1 静态。
 * 评估仍走 eval_apply(worker 按 zl->d_in 自动 φ 提升), ER 恒对原始 R。 */
static void run_map_arm(const char *arm, const float *XF, int din, const float *bc,
                        float tr, const float *Reff, const float *R, const float *X,
                        const float *Ys, const float *Yt_ev, const float *wv,
                        const float *swts, /* 感知加权解算: 靶按 √cls权 白化, NULL=关 */
                        const int *fit, int nf, const int *ev, const int *mode0, int nev,
                        const int *ranks, int nrank, const double *lambdas, int nlam,
                        int maxk, const ds4_loss_weights *lw, double dscale,
                        uint64_t seed, int nth, int L, FILE *lf, best_t *best,
                        float *Yhat, float *Cb, float *Cp, float *zcat) {
    float *Xm = xmalloc((size_t)nf * din * sizeof(float));
    float *Rm = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        memcpy(Xm + (size_t)i * din, XF + (size_t)fit[i] * din, (size_t)din * sizeof(float));
        const float *rs = Reff + (size_t)fit[i] * D;
        float *rd = Rm + (size_t)i * D;
        if (swts) for (int j = 0; j < D; j++) rd[j] = rs[j] * swts[j];
        else memcpy(rd, rs, D * sizeof(float));
    }
    int nfa = nf;
    if (g_smooth_aug) {   /* L_smooth 进解算目标: 同靶扰动增广行(逼 M(x+δ)≈M(x)),
                           * δ=ds4_loss_dither 定死种子(行号=fit 行, 与 held 评估行不交) */
        Xm = realloc(Xm, (size_t)2 * nf * din * sizeof(float));
        Rm = realloc(Rm, (size_t)2 * nf * D * sizeof(float));
        if (!Xm || !Rm) die("L%d %s 增广 OOM", L, arm);
        float *dl = xmalloc(D * sizeof(float)), *xp = xmalloc(D * sizeof(float));
        for (int i = 0; i < nf; i++) {
            const float *x = X + (size_t)fit[i] * D;
            double ss = 0; for (int j = 0; j < D; j++) ss += (double)x[j] * x[j];
            ds4_loss_dither(seed, (uint32_t)fit[i], (float)(dscale * sqrt(ss / D)), dl, D);
            for (int j = 0; j < D; j++) xp[j] = x[j] + dl[j];
            float *o = Xm + (size_t)(nf + i) * din;
            if (din == 3 * D) mk_phi(xp, o);
            else memcpy(o, xp, D * sizeof(float));
            memcpy(Rm + (size_t)(nf + i) * D, Rm + (size_t)i * D, D * sizeof(float));
        }
        free(dl); free(xp);
        nfa = 2 * nf;
    }
    ds4_z *zs[MAXG] = {0};
    float lamf[MAXG];
    for (int li = 0; li < nlam; li++) lamf[li] = (float)lambdas[li];
    if (ds4_z_solve_multi(Xm, Rm, (uint32_t)nfa, (uint32_t)din, D, (uint32_t)maxk,
                          lamf, (uint32_t)nlam, zs))
        die("L%d %s 解算失败", L, arm);
    if (swts)   /* 解在白化空间, U 行回真空间(秩截断已按 cls 权分配容量) */
        for (int li = 0; li < nlam; li++)
            for (uint32_t j = 0; j < (uint32_t)D; j++) {
                float inv = swts[j] > 1e-20f ? 1.0f / swts[j] : 0.0f;
                for (uint32_t c = 0; c < zs[li]->rank; c++)
                    zs[li]->U[(size_t)j * zs[li]->rank + c] *= inv;
            }
    for (int li = 0; li < nlam; li++) {
        ds4_z *zl[1] = { zs[li] };
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
                if (g_track_z) {         /* 图臂夺冠: 截秩克隆(swts 已回真空间) */
                    if (best->zkeep) ds4_z_free(best->zkeep);
                    best->zkeep = z_clone_k(zl[0]);
                    best->zkeep_hasge = (bc != NULL);
                    best->bckind = 0;
                }
            }
            fflush(stdout);
        }
        ds4_z_free(zl[0]);
    }
    free(Xm); free(Rm);
}

/* GE 单臂评估: 图=空, bc=GE 修正 */
/* 纯基修正臂(无图): GE / 乘性出口 / 其叠加 —— bc 即全部修正 */
static void run_bc_arm(const char *arm, const float *params, int nparam,
                       const float *bc, const float *X,
                       const float *Ys, const float *Yt_ev, const float *wv,
                       const float *R, const int *ev, const int *mode0, int nev,
                       const ds4_loss_weights *lw, double dscale, uint64_t seed,
                       int nth, int L, FILE *lf, best_t *best,
                       float *Yhat, float *Cb, float *Cp) {
    ds4_z *zl[1] = {0};
    double er = eval_apply(zl, NULL, 1, 1 /*路由门语义: 扰动不改基修正*/, bc, 0,
                           X, Ys, R, ev, mode0, nev, Yhat, Cb, Cp, dscale, seed, nth);
    float la = ds4_loss_align(Yhat, Yt_ev, (uint32_t)nev, D);
    float lc = ds4_loss_classify(Yhat, Yt_ev, wv, (uint32_t)nev, D);
    float ls = ds4_loss_smooth(Cb, Cp, (uint32_t)nev, D);
    float lfx = ds4_loss_fixed(params, (size_t)nparam);
    float tot = ds4_loss_total(lw, la, lc, ls, lfx);
    int nact = 0;
    for (int e = 0; e < nparam; e++) if (fabsf(params[e]) > 1e-4f) nact++;
    fprintf(lf, "%s 1 0 0 %.6f %.6f %.6f %.6f %.6f %.2f %.2f\n",
            arm, la, lc, ls, lfx, tot, er * 100, nparam * 2 / 1e6);
    printf("  L%d %-6s 活参=%-4d    | align %.4f cls %.4f sm %.5f fx %.5f "
           "tot %.4f | ER %.1f%% | vol %.2fMB\n", L, arm, nact, la, lc, ls, lfx, tot,
           er * 100, nparam * 2 / 1e6);
    if (tot < best->tot) {
        best->tot = tot; best->lam = 0; best->k = 0; best->M = 1;
        best->er = er; best->la = la; best->lc = lc; best->arm = arm;
        if (g_track_z) {                 /* 基修正臂夺冠: z 克隆作废, 记门参数 */
            if (best->zkeep) { ds4_z_free(best->zkeep); best->zkeep = NULL; }
            best->zkeep_hasge = 0;
            if (nparam == 256) { memcpy(best->bcdz, params, 256 * sizeof(float)); best->bckind = 1; }
            else best->bckind = 2;       /* mul/GEc 等: 无引擎序列化路, 落地时响亮报 */
        }
    }
    fflush(stdout);
}

/* 乘性出口臂: 每通道增益 ua_j(引擎 zc_zl_add_kernel mul 分支 ⊙(1+ua) 同式),
 * 直击巨值通道的通道级系统偏差(L41 定向: 裸 cls=L20 的 3×=通道指纹)。
 * 闭式=每通道标量 ridge: ua_j = Σ_fit R_j·base_j / (Σ_fit base_j² + λ·量纲)。 */
static float *solve_mul(const float *R, const float *base, const uint8_t *isfit,
                        int ntok, double lam) {
    double *num = xmalloc(D * sizeof(double)), *den = xmalloc(D * sizeof(double));
    memset(num, 0, D * sizeof(double)); memset(den, 0, D * sizeof(double));
    for (int t = 0; t < ntok; t++) {
        if (!isfit[t]) continue;
        const float *r = R + (size_t)t * D, *b = base + (size_t)t * D;
        for (int j = 0; j < D; j++) { num[j] += (double)r[j] * b[j]; den[j] += (double)b[j] * b[j]; }
    }
    double md = 0; for (int j = 0; j < D; j++) md += den[j];
    md = md / D + 1e-30;
    float *ua = xmalloc(D * sizeof(float));
    for (int j = 0; j < D; j++) ua[j] = (float)(num[j] / (den[j] + lam * md));
    free(num); free(den);
    return ua;
}
static float *mul_corr_build(const float *ua, const float *base, const float *bc0, int ntok) {
    float *corr = xmalloc((size_t)ntok * D * sizeof(float));
    for (int t = 0; t < ntok; t++) {
        const float *b = base + (size_t)t * D;
        const float *c0 = bc0 ? bc0 + (size_t)t * D : NULL;
        float *c = corr + (size_t)t * D;
        for (int j = 0; j < D; j++) c[j] = ua[j] * b[j] + (c0 ? c0[j] : 0.0f);
    }
    return corr;
}

/* ---- zcache 读取: R=dH, Ys=Σ pw·pYQ(自洽重建, 不掺 teacher_routed 的约定差) ---- */
static void zcache_load(const char *dir, int L, int ntok, float **R_out, float **Ys_out,
                        zpairs *zp) {
    char p[1024]; snprintf(p, sizeof p, "%s/zcache_L%02d.npz", dir, L);
    FILE *f = fopen(p, "rb");
    if (!f) die("%s 打不开(先 DS4_ZL_CACHE_ONLY=1 跑 zlayer 建缓存)", p);
    fseeko(f, 0, SEEK_END); long long sz = (long long)ftello(f); fseeko(f, 0, SEEK_SET);
    uint8_t *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("%s 读不满", p);
    fclose(f);
    /* 大条目(dH/pYQ) f32 直读, 绕开 f64 中转的 2× 内存搬运(提速轮) */
    float *R = NULL, *pyq = NULL; int64_t rd0, rd1, q0, q1;
    npz_arr prow, pe, pw;
    if (npz_get_f32(buf, sz, "dH", &R, &rd0, &rd1) || npz_get(buf, sz, "prow", &prow) ||
        npz_get(buf, sz, "pe", &pe) || npz_get(buf, sz, "pw", &pw) ||
        npz_get_f32(buf, sz, "pYQ", &pyq, &q0, &q1)) die("%s 字段缺(要 dH/prow/pe/pw/pYQ)", p);
    if (rd0 < ntok || rd1 != D)
        die("zcache dH 形状 %lldx%lld, 需 ≥%dx%d", (long long)rd0, (long long)rd1, ntok, D);
    if (q1 != D || q0 != prow.n || pw.n != prow.n || pe.n != prow.n)
        die("zcache 配对形状不一致: prow=%lld pYQ=%lldx%lld", (long long)prow.n,
            (long long)q0, (long long)q1);
    float *Ys = xmalloc((size_t)ntok * D * sizeof(float));
    memset(Ys, 0, (size_t)ntok * D * sizeof(float));
    zp->npair = prow.n;                  /* 配对留给 GE 臂(路由精确条件化数据面) */
    zp->prow = xmalloc((size_t)prow.n * sizeof(int));
    zp->pe = xmalloc((size_t)prow.n * sizeof(int));
    zp->pw = xmalloc((size_t)prow.n * sizeof(float));
    zp->pyq = pyq;
    for (long long i = 0; i < prow.n; i++) {
        int t = (int)prow.v[i], e = (int)pe.v[i];
        if (t < 0 || e < 0) die("zcache 配对越界: prow=%d pe=%d", t, e);
        zp->prow[i] = t; zp->pe[i] = e; zp->pw[i] = (float)pw.v[i];
        if (t >= ntok) continue;
        float w = zp->pw[i];
        float *ys = Ys + (size_t)t * D;
        const float *yq = pyq + (size_t)i * D;
        for (int d = 0; d < D; d++) ys[d] += w * yq[d];
    }
    free(prow.v); free(pe.v); free(pw.v); free(buf);
    *R_out = R; *Ys_out = Ys;
}

/* cls 方差权²(fit 侧教师方差归一, ftAw/量化侧 DS4_TUNE 同精神; held 不参与=防泄漏) */
static float *mk_sw2_fit(const float *R, const float *Ys, const int *fit, int nf) {
    float *Ytf = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        const float *ys = Ys + (size_t)fit[i] * D, *r = R + (size_t)fit[i] * D;
        float *o = Ytf + (size_t)i * D;
        for (int j = 0; j < D; j++) o[j] = ys[j] + r[j];
    }
    float *wvf = xmalloc(D * sizeof(float));
    ds4_loss_dim_variance(Ytf, (uint32_t)nf, D, wvf);
    free(Ytf);
    double mw = 0; for (int j = 0; j < D; j++) mw += wvf[j];
    mw = mw / D + 1e-30;
    float *sw2 = xmalloc(D * sizeof(float));
    for (int j = 0; j < D; j++) sw2[j] = (float)((wvf[j] + 1e-6 * mw) / mw);
    free(wvf);
    return sw2;
}

/* 基修正臂组: GE → 乘性出口 → GE+乘性; 产出 gecorr 与 Rge(=R−GE, ftA 叠加臂复用) */
static void run_base_arms(const zpairs *zp, const float *X, const float *R,
                          const float *Ys, const float *Yt_ev, const float *wv,
                          const int *fit, int nf, const int *ev, const int *mode0,
                          int nev, int ntok, const ds4_loss_weights *lw, double dscale,
                          uint64_t seed, int nth, int L, FILE *lf, best_t *best,
                          float *Yhat, float *Cb, float *Cp,
                          float **gecorr_out, float **Rge_out, uint8_t **isfit_out,
                          float *dz_out) {
    uint8_t *isfit = xmalloc((size_t)ntok);
    memset(isfit, 0, (size_t)ntok);
    for (int i = 0; i < nf; i++) isfit[fit[i]] = 1;
    double delta[256]; float dzf[256];
    solve_ge(zp, R, isfit, ntok, 1e-3, L, delta, NULL);  /* λ=zlayer GE_LAM 同款 */
    float *gecorr = ge_corr_build(zp, delta, ntok);
    for (int e = 0; e < 256; e++) dzf[e] = (float)delta[e];
    if (dz_out) memcpy(dz_out, dzf, 256 * sizeof(float));
    run_bc_arm("GE", dzf, 256, gecorr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
               lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
    {   /* GEw 臂: cls 方差权(fit 侧教师, 防 held 泄漏)白化进 GE 解算目标 —— 四损失
         * L_classify 从裁判升级为目标组件(用户令); 评估仍在真空间, 四损失总分裁决。 */
        float *sw2 = mk_sw2_fit(R, Ys, fit, nf);
        double dw[256]; float dwf[256];
        solve_ge(zp, R, isfit, ntok, 1e-3, L, dw, sw2);
        float *gwcorr = ge_corr_build(zp, dw, ntok);
        for (int e = 0; e < 256; e++) dwf[e] = (float)dw[e];
        run_bc_arm("GEw", dwf, 256, gwcorr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
                   lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
        free(gwcorr); free(sw2);
    }
    float *ua = solve_mul(R, Ys, isfit, ntok, 1e-3);
    float *mc = mul_corr_build(ua, Ys, NULL, ntok);
    run_bc_arm("mul", ua, D, mc, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
               lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
    free(mc);
    float *Rge = xmalloc((size_t)ntok * D * sizeof(float));
    float *base2 = xmalloc((size_t)ntok * D * sizeof(float));
    for (size_t i = 0; i < (size_t)ntok * D; i++) {
        Rge[i] = R[i] - gecorr[i];
        base2[i] = Ys[i] + gecorr[i];      /* 引擎里 GE 折进权重 → 乘门作用在 GE 后 */
    }
    float *ua2 = solve_mul(Rge, base2, isfit, ntok, 1e-3);
    float *mc2 = mul_corr_build(ua2, base2, gecorr, ntok);
    run_bc_arm("GE+mul", ua2, D, mc2, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
               lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
    /* 折半稳定性诊断("没找到≠没有"的定量化): fit 交替对半双解, 两组参数余弦。
     * 高相关+输 held = 模型家族错(结构在, 形态不对); 低相关 = 数据饿
     * (每专家/每通道有效样本不足, 解在拟合噪声)。 */
    {
        uint8_t *h1 = xmalloc((size_t)ntok), *h2 = xmalloc((size_t)ntok);
        memset(h1, 0, (size_t)ntok); memset(h2, 0, (size_t)ntok);
        for (int i = 0; i < nf; i++) { if (i & 1) h2[fit[i]] = 1; else h1[fit[i]] = 1; }
        double da[256], db[256];
        solve_ge(zp, R, h1, ntok, 1e-3, L, da, NULL);
        solve_ge(zp, R, h2, ntok, 1e-3, L, db, NULL);
        double nn = 0, na = 0, nb = 0;
        for (int e = 0; e < 256; e++) { nn += da[e] * db[e]; na += da[e] * da[e]; nb += db[e] * db[e]; }
        double cge = nn / (sqrt(na * nb) + 1e-30);
        float *u1 = solve_mul(R, Ys, h1, ntok, 1e-3), *u2 = solve_mul(R, Ys, h2, ntok, 1e-3);
        nn = na = nb = 0;
        for (int j = 0; j < D; j++) { nn += (double)u1[j] * u2[j]; na += (double)u1[j] * u1[j]; nb += (double)u2[j] * u2[j]; }
        double cmu = nn / (sqrt(na * nb) + 1e-30);
        /* fit↔held δ 余弦: 折半稳定但此值塌 = 参数随语料段漂移(语境条件化结构) */
        uint8_t *hv = xmalloc((size_t)ntok);
        memset(hv, 0, (size_t)ntok);
        for (int i = 0; i < nev; i++) hv[ev[i]] = 1;
        double dv[256];
        solve_ge(zp, R, hv, ntok, 1e-3, L, dv, NULL);
        nn = na = nb = 0;
        for (int e = 0; e < 256; e++) { nn += delta[e] * dv[e]; na += delta[e] * delta[e]; nb += dv[e] * dv[e]; }
        printf("  L%d 折半稳定性: GEδ cos=%.3f  乘门ua cos=%.3f  fit↔held δ cos=%.3f "
               "(折半高+此值塌=段漂移)\n", L, cge, cmu, nn / (sqrt(na * nb) + 1e-30));
        free(h1); free(h2); free(u1); free(u2); free(hv);
    }
    free(ua); free(ua2); free(mc2); free(base2);
    *gecorr_out = gecorr; *Rge_out = Rge; *isfit_out = isfit;   /* 语境门(GEc)复用 */
}
