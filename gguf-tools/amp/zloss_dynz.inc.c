/* zloss_dynz.inc.c — ★方案B 动态 z(用户核心设计, 2026-08-26 用户令一层证明)★
 * (只被 zloss_solve.c include)
 *
 * 设计(amp_campaign.sh 头注释的用户原式, 一字不改):
 *   pv_c(x) = tanh(V_c·x/s_c) · tanh(A_c·x/σ_c)   — 两 tanh 乘积 = 真二阶门,
 *   修正 y = U · pv(x)                              — 对 U 仍线性 ⇒ 闭式 ridge, 零训练。
 * z 是 x 的函数(每 token 的门控隐变量), 不是静态参数 —— "低维动态 z 隐射高维行为"。
 *
 * 方向来源(零训练闭式): V = 线性解(ds4_z_solve)的输入方向(读主结构);
 * A = 线性修正后残差再解一遍的输入方向(读线性读不出的部分); s/σ = fit 侧投影 std
 * (tanh 工作区 ±1)。判决=四损失 held(run_bc_arm 同路, 与静态臂同表同尺)。
 * 本针 smooth 项: bc 臂路不建模扰动响应(GE 同待遇), align/cls 主导判决 — 响亮声明。
 */

/* Φ[t,c] = tanh((x_t·V_c)/s_c)·tanh((x_t·A_c)/σ_c), 投影用 BLAS 批量 */
static float *dynz_phi(const float *X, int ntok, const float *V, const float *A,
                       const float *sv, const float *sa, int k, int gate2) {
    float *p1 = xmalloc((size_t)ntok * k * sizeof(float));
    float *p2 = xmalloc((size_t)ntok * k * sizeof(float));
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ntok, k, D,
                1.0f, X, D, V, k, 0.0f, p1, k);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ntok, k, D,
                1.0f, X, D, A, k, 0.0f, p2, k);
#else
    for (int t = 0; t < ntok; t++) for (int c = 0; c < k; c++) {
        double a1 = 0, a2 = 0;
        for (int j = 0; j < D; j++) {
            a1 += (double)X[(size_t)t * D + j] * V[(size_t)j * k + c];
            a2 += (double)X[(size_t)t * D + j] * A[(size_t)j * k + c];
        }
        p1[(size_t)t * k + c] = (float)a1; p2[(size_t)t * k + c] = (float)a2;
    }
#endif
    for (size_t i = 0; i < (size_t)ntok * k; i++) {
        int c = (int)(i % k);
        float a = p1[i] / sv[c], b = p2[i] / sa[c];
        /* 0=单门 tanh(a)  1=双门 tanh(a)·tanh(b)  2=线性×门 a·tanh(b):
         * 门饱和→±线性 ⇒ 下界=静态线性臂(L10 自检: tanh 压扁大幅值投影=丢巨值矿) */
        p1[i] = gate2 == 1 ? tanhf(a) * tanhf(b) : gate2 == 2 ? a * tanhf(b) : tanhf(a);
    }
    free(p2);
    return p1;                                     /* p1 即 Φ */
}

/* 方案B 臂: 方向两遍闭式 → (k,λ) 网格 U ridge → 全行修正 → run_bc_arm 四损失判 */
static void run_dynz_arm(const float *X, const float *R, const float *Ys,
                         const float *Yt_ev, const float *wv, const int *fit, int nf,
                         const int *ev, const int *mode0, int nev, int ntok,
                         const int *ranks, int nrank, const double *lambdas, int nlam,
                         int maxk, const ds4_loss_weights *lw, double dscale,
                         uint64_t seed, int nth, int L, FILE *lf, best_t *best,
                         float *Yhat, float *Cb, float *Cp) {
    double t0 = tnow();
    float *Xf = xmalloc((size_t)nf * D * sizeof(float));
    float *Rf = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        memcpy(Xf + (size_t)i * D, X + (size_t)fit[i] * D, D * sizeof(float));
        memcpy(Rf + (size_t)i * D, R + (size_t)fit[i] * D, D * sizeof(float));
    }
    float lam0 = 3e-2f;
    ds4_z *s1 = ds4_z_solve(Xf, Rf, (uint32_t)nf, D, D, (uint32_t)maxk, lam0);
    if (!s1) die("L%d dynz 第一遍方向解失败", L);
    /* 线性修正后的残差 R2 = Rf − (Xf·V1)·diag(z)·U1ᵀ (BLAS 两 GEMM) */
    float *pv = xmalloc((size_t)nf * maxk * sizeof(float));
    float *R2 = xmalloc((size_t)nf * D * sizeof(float));
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nf, maxk, D,
                1.0f, Xf, D, s1->V, s1->rank, 0.0f, pv, maxk);
    for (size_t i = 0; i < (size_t)nf * maxk; i++) pv[i] *= s1->z[i % maxk];
    memcpy(R2, Rf, (size_t)nf * D * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nf, D, maxk,
                -1.0f, pv, maxk, s1->U, s1->rank, 1.0f, R2, D);
#else
    for (int i = 0; i < nf; i++) {                 /* 标量回落: apply 后相减 */
        float *y = R2 + (size_t)i * D;
        ds4_z_apply(s1, Xf + (size_t)i * D, y);
        const float *r = Rf + (size_t)i * D;
        for (int j = 0; j < D; j++) y[j] = r[j] - y[j];
    }
    (void)pv;
#endif
    ds4_z *s2 = ds4_z_solve(Xf, R2, (uint32_t)nf, D, D, (uint32_t)maxk, lam0);
    if (!s2) die("L%d dynz 第二遍方向解失败", L);
    free(pv); free(R2); free(Xf); free(Rf);
    /* 投影尺度 s/σ = fit 侧 std(V_c·x) — tanh 工作区标定, held 不参与(防泄漏) */
    float *sv = xmalloc(maxk * sizeof(float)), *sa = xmalloc(maxk * sizeof(float));
    {
        float *Pf = xmalloc((size_t)nf * maxk * sizeof(float));
        for (int pass = 0; pass < 2; pass++) {
            const ds4_z *s = pass ? s2 : s1; float *out = pass ? sa : sv;
#ifdef DQ_BLAS
            float *Xf2 = xmalloc((size_t)nf * D * sizeof(float));
            for (int i = 0; i < nf; i++) memcpy(Xf2 + (size_t)i * D, X + (size_t)fit[i] * D, D * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nf, maxk, D,
                        1.0f, Xf2, D, s->V, s->rank, 0.0f, Pf, maxk);
            free(Xf2);
#else
            for (int i = 0; i < nf; i++) for (int c = 0; c < maxk; c++) {
                double a = 0; const float *x = X + (size_t)fit[i] * D;
                for (int j = 0; j < D; j++) a += (double)x[j] * s->V[(size_t)j * s->rank + c];
                Pf[(size_t)i * maxk + c] = (float)a;
            }
#endif
            for (int c = 0; c < maxk; c++) {
                double ss = 0;
                for (int i = 0; i < nf; i++) { double v = Pf[(size_t)i * maxk + c]; ss += v * v; }
                out[c] = (float)sqrt(ss / nf) + 1e-6f;
            }
        }
        free(Pf);
    }
    fprintf(stderr, "  [dynz]L%d 方向双遍 %.0fs (V=主结构 A=线性残差方向)\n", L, tnow() - t0);
    /* (k,λ) 网格: V/A 取前 k 列(奇异值降序=最优截断), U 闭式 ridge */
    for (int ki = 0; ki < nrank; ki++) {
        int k = ranks[ki]; if (k > maxk) continue;
        float *Vk = xmalloc((size_t)D * k * sizeof(float)), *Ak = xmalloc((size_t)D * k * sizeof(float));
        for (int j = 0; j < D; j++) {
            memcpy(Vk + (size_t)j * k, s1->V + (size_t)j * s1->rank, (size_t)k * sizeof(float));
            memcpy(Ak + (size_t)j * k, s2->V + (size_t)j * s2->rank, (size_t)k * sizeof(float));
        }
        for (int gate2 = 0; gate2 <= 2; gate2++) {
        float *Phi = dynz_phi(X, ntok, Vk, Ak, sv, sa, k, gate2);   /* 全行(fit 解, held 判) */
        double *G = xmalloc((size_t)k * k * sizeof(double));    /* Gram=ΦfᵀΦf */
        double *B = xmalloc((size_t)k * D * sizeof(double));    /* ΦfᵀR */
        memset(G, 0, (size_t)k * k * sizeof(double));
        memset(B, 0, (size_t)k * D * sizeof(double));
        for (int i = 0; i < nf; i++) {
            const float *ph = Phi + (size_t)fit[i] * k, *r = R + (size_t)fit[i] * D;
            for (int c = 0; c < k; c++) {
                double pc = ph[c];
                for (int c2 = c; c2 < k; c2++) G[(size_t)c * k + c2] += pc * ph[c2];
                for (int j = 0; j < D; j++) B[(size_t)c * D + j] += pc * r[j];
            }
        }
        for (int c = 0; c < k; c++) for (int c2 = 0; c2 < c; c2++)
            G[(size_t)c * k + c2] = G[(size_t)c2 * k + c];
        double tr = 0; for (int c = 0; c < k; c++) tr += G[(size_t)c * k + c];
        tr = tr / k + 1e-30;
        for (int li = 0; li < nlam; li++) {
            double *Gs = xmalloc((size_t)k * k * sizeof(double));
            double *Us = xmalloc((size_t)k * D * sizeof(double));
            memcpy(Gs, G, (size_t)k * k * sizeof(double));
            memcpy(Us, B, (size_t)k * D * sizeof(double));
            for (int c = 0; c < k; c++) Gs[(size_t)c * k + c] += lambdas[li] * tr;
            if (chol_solve_spd(Gs, k, Us, D) != 0) { free(Gs); free(Us); continue; }
            float *corr = xmalloc((size_t)ntok * D * sizeof(float));   /* Φ·U (U=Us[k,D]) */
#ifdef DQ_BLAS
            float *Uf = xmalloc((size_t)k * D * sizeof(float));
            for (size_t i = 0; i < (size_t)k * D; i++) Uf[i] = (float)Us[i];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ntok, D, k,
                        1.0f, Phi, k, Uf, D, 0.0f, corr, D);
            free(Uf);
#else
            for (int t = 0; t < ntok; t++) for (int j = 0; j < D; j++) {
                double a = 0; const float *ph = Phi + (size_t)t * k;
                for (int c = 0; c < k; c++) a += (double)ph[c] * Us[(size_t)c * D + j];
                corr[(size_t)t * D + j] = (float)a;
            }
#endif
            float *Uf2 = xmalloc((size_t)k * D * sizeof(float));
            for (size_t i = 0; i < (size_t)k * D; i++) Uf2[i] = (float)Us[i];
            double fn2 = 0, fd2 = 0;      /* fit 侧挽回(实现自检: fit 也没肉=解算病) */
            for (int i = 0; i < nf; i += 7) {
                const float *r = R + (size_t)fit[i] * D, *c2 = corr + (size_t)fit[i] * D;
                for (int j = 0; j < D; j++) { double e = (double)r[j] - c2[j];
                    fn2 += e * e; fd2 += (double)r[j] * r[j]; }
            }
            const char *anm = gate2 == 1 ? "dynz" : gate2 == 2 ? "dynL" : "dyn1";
            run_bc_arm(anm, Uf2, k * D, corr, X, Ys, Yt_ev, wv,
                       R, ev, mode0, nev, lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
            printf("    ↑%s k=%d λ=%.3g fitER=%.1f%% vol=%.1fMB\n", anm,
                   k, lambdas[li], 100.0 * (1.0 - fn2 / (fd2 + 1e-30)),
                   (double)k * (gate2 ? 3 : 2) * D * 2 / 1e6);
            free(Uf2); free(corr); free(Gs); free(Us);
        }
        free(Phi); free(G); free(B);
        }
        free(Vk); free(Ak);
    }
    ds4_z_free(s1); ds4_z_free(s2); free(sv); free(sa);
}

/* ★pez 臂: 每专家低秩 z(路由承载的动态 z)★ — GE 标量门的向量化:
 *   corr_t = Σ_{e∈S_t} w_te · U_e · (Vᵀ x_t),  V 共享(线性解方向), U_e 每专家独立。
 * 门=路由本身(部署免费, 不用学) ⇒ z 经由路由成为 x 的函数。特征 φ_{(e,c)}(t)=
 * w_te·p_c(t)(e 未点火=0), U 联合 ridge(256k 维块正规方程, 闭式)。
 * 体积: U[256,k,D]+V[D,k] f16 ≈ k=16 时 33.6MB/层。 */
static void run_pez_arm(const zpairs *zp, const float *X, const float *R, const float *Ys,
                        const float *Yt_ev, const float *wv, const int *fit, int nf,
                        const int *ev, const int *mode0, int nev, int ntok,
                        const double *lambdas, int nlam, int maxk,
                        const ds4_loss_weights *lw, double dscale, uint64_t seed,
                        int nth, int L, FILE *lf, best_t *best,
                        float *Yhat, float *Cb, float *Cp) {
    double t0 = tnow();
    float *Xf = xmalloc((size_t)nf * D * sizeof(float));
    float *Rf = xmalloc((size_t)nf * D * sizeof(float));
    for (int i = 0; i < nf; i++) {
        memcpy(Xf + (size_t)i * D, X + (size_t)fit[i] * D, D * sizeof(float));
        memcpy(Rf + (size_t)i * D, R + (size_t)fit[i] * D, D * sizeof(float));
    }
    ds4_z *sv1 = ds4_z_solve(Xf, Rf, (uint32_t)nf, D, D, (uint32_t)maxk, 3e-2f);
    if (!sv1) die("L%d pez 共享 V 解失败", L);
    free(Xf); free(Rf);
    uint8_t *isfit = xmalloc((size_t)ntok); memset(isfit, 0, (size_t)ntok);
    for (int i = 0; i < nf; i++) isfit[fit[i]] = 1;
    /* 行→配对索引(zcache 的 prow 有序? 不假定, 建行首索引) */
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
    int kk[3] = {8, 16, 32};
    for (int ki = 0; ki < 3; ki++) {
        int k = kk[ki]; if (k > maxk) continue;
        int F = 256 * k;
        /* p[t,c] = x_t·V_c / s_c(sv1 前 k 列; 尺度并进特征保 Gram 条件) */
        float *pm = xmalloc((size_t)ntok * k * sizeof(float));
        {
            float *Vk = xmalloc((size_t)D * k * sizeof(float));
            for (int j = 0; j < D; j++)
                memcpy(Vk + (size_t)j * k, sv1->V + (size_t)j * sv1->rank, (size_t)k * sizeof(float));
#ifdef DQ_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ntok, k, D,
                        1.0f, X, D, Vk, k, 0.0f, pm, k);
#else
            for (int t = 0; t < ntok; t++) for (int c = 0; c < k; c++) {
                double a = 0; for (int j = 0; j < D; j++) a += (double)X[(size_t)t*D+j]*Vk[(size_t)j*k+c];
                pm[(size_t)t * k + c] = (float)a;
            }
#endif
            free(Vk);
            for (int c = 0; c < k; c++) {
                double ss = 0; int n2 = 0;
                for (int i = 0; i < nf; i++) { double v = pm[(size_t)fit[i]*k+c]; ss += v*v; n2++; }
                float sc2 = (float)sqrt(ss / (n2 ? n2 : 1)) + 1e-6f;
                for (int t = 0; t < ntok; t++) pm[(size_t)t * k + c] /= sc2;
            }
        }
        /* Gram[F,F] 与 B[F,D](fit 行): 每行活跃特征 ≤ NACT_MAX·k */
        double *G = xmalloc((size_t)F * F * sizeof(double));
        double *B = xmalloc((size_t)F * D * sizeof(double));
        memset(G, 0, (size_t)F * F * sizeof(double));
        memset(B, 0, (size_t)F * D * sizeof(double));
        int *fidx = xmalloc(64 * k * sizeof(int));
        float *fval = xmalloc(64 * k * sizeof(float));
        for (int t = 0; t < ntok; t++) {
            if (!isfit[t]) continue;
            int na = 0;
            for (long long a = rs[t]; a < rs[t + 1] && na < 64 * k; a++) {
                long long i = ord[a]; int e = zp->pe[i]; float w = zp->pw[i];
                for (int c = 0; c < k; c++) { fidx[na] = e * k + c; fval[na] = w * pm[(size_t)t*k+c]; na++; }
            }
            const float *r = R + (size_t)t * D;
            for (int a = 0; a < na; a++) {
                double va = fval[a]; int fa = fidx[a];
                for (int b = a; b < na; b++) {
                    double d = va * fval[b];
                    G[(size_t)fa * F + fidx[b]] += d;
                    if (fidx[b] != fa) G[(size_t)fidx[b] * F + fa] += d;
                }
                double *Br = B + (size_t)fa * D;
                for (int j = 0; j < D; j++) Br[j] += va * r[j];
            }
        }
        double tr = 0; for (int f2 = 0; f2 < F; f2++) tr += G[(size_t)f2 * F + f2];
        tr = tr / F + 1e-30;
        for (int li = 0; li < nlam; li++) {
            double *Gs = xmalloc((size_t)F * F * sizeof(double));
            double *Us = xmalloc((size_t)F * D * sizeof(double));
            memcpy(Gs, G, (size_t)F * F * sizeof(double));
            memcpy(Us, B, (size_t)F * D * sizeof(double));
            for (int f2 = 0; f2 < F; f2++) Gs[(size_t)f2 * F + f2] += lambdas[li] * tr;
            if (chol_solve_spd(Gs, F, Us, D) != 0) { free(Gs); free(Us); continue; }
            float *corr = xmalloc((size_t)ntok * D * sizeof(float));
            memset(corr, 0, (size_t)ntok * D * sizeof(float));
            double fn2 = 0, fd2 = 0;
            for (int t = 0; t < ntok; t++) {
                float *ct = corr + (size_t)t * D;
                for (long long a = rs[t]; a < rs[t + 1]; a++) {
                    long long i = ord[a]; int e = zp->pe[i]; double w = zp->pw[i];
                    for (int c = 0; c < k; c++) {
                        double f3 = w * pm[(size_t)t * k + c];
                        if (fabs(f3) < 1e-12) continue;
                        const double *ur = Us + (size_t)(e * k + c) * D;
                        for (int j = 0; j < D; j++) ct[j] += (float)(f3 * ur[j]);
                    }
                }
                if (isfit[t] && (t % 7) == 0) {
                    const float *r = R + (size_t)t * D;
                    for (int j = 0; j < D; j++) { double e2 = (double)r[j] - ct[j];
                        fn2 += e2 * e2; fd2 += (double)r[j] * r[j]; }
                }
            }
            float *Uf2 = xmalloc((size_t)F * D * sizeof(float));
            for (size_t i = 0; i < (size_t)F * D; i++) Uf2[i] = (float)Us[i];
            run_bc_arm("pez", Uf2, F * D, corr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
                       lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
            printf("    ↑pez k=%d λ=%.3g fitER=%.1f%% vol=%.1fMB(U[256,%d,D]+V f16)\n",
                   k, lambdas[li], 100.0 * (1.0 - fn2 / (fd2 + 1e-30)),
                   ((double)256 * k * D + (double)D * k) * 2 / 1e6, k);
            free(Uf2); free(corr); free(Gs); free(Us);
        }
        free(pm); free(G); free(B); free(fidx); free(fval);
    }
    ds4_z_free(sv1); free(isfit); free(rs); free(ord);
    fprintf(stderr, "  [pez]L%d 每专家低秩 z 收官 %.0fs\n", L, tnow() - t0);
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


/* ★4L 臂(用户设计: 四损失全部进解算目标, 机器=ds4_z 原样)★
 * align   → 行权 1/‖Yt_t‖(方向对齐, token 平权 — 能量口径的"修残差"按范数分配容量,
 *           大范数行霸占解, 这正是用户判"修残差挖不到"的病灶);
 * classify→ 列 √方差白化(解在白化空间, run_map_arm 内 U 行回真空间);
 * smooth  → --smooth-aug 扰动增广行(调用方开);
 * fixed   → λ 网格(量纲化 ridge)。 */
static void run_4l_arm(const float *X, const float *R, const float *Ys, const float *Yt,
                       const float *Yt_ev, const float *wv, const int *fit, int nf,
                       const int *ev, const int *mode0, int nev, int ntok,
                       const int *ranks, int nrank, const double *lambdas, int nlam,
                       int maxk, const ds4_loss_weights *lw, double dscale, float tr,
                       uint64_t seed, int nth, int L, FILE *lf, best_t *best,
                       float *Yhat, float *Cb, float *Cp, float *zcat,
                       const float *gecorr, const float *Rge) {
    float *sw2 = mk_sw2_fit(R, Ys, fit, nf);
    float *swts = xmalloc(D * sizeof(float));
    for (int j = 0; j < D; j++) swts[j] = sqrtf(sw2[j]);
    float *ralign = xmalloc((size_t)ntok * sizeof(float));
    double mrw = 0;
    for (int t = 0; t < ntok; t++) {
        const float *yt = Yt + (size_t)t * D;
        double nn = 0; for (int j = 0; j < D; j++) nn += (double)yt[j] * yt[j];
        ralign[t] = (float)(1.0 / (sqrt(nn) + 1e-9));
        mrw += ralign[t];
    }
    mrw /= ntok;   /* 行权均值归一: λ 量纲与非加权臂可比 */
    for (int t = 0; t < ntok; t++) ralign[t] /= (float)mrw;
    run_map_arm("4L", X, D, NULL, tr, R, R, X, Ys, Yt_ev, wv, swts, ralign,
                fit, nf, ev, mode0, nev, ranks, nrank, lambdas, nlam, maxk,
                lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp, zcat);
    /* ★GE 基版本(2026-08-27): 无此臂时择优只能在"纯 z"与"GE"间二选一, 硬凑叠加
     * = z 修全 R 而 GE 又修一遍同一部分 → 重复修正踩重尾(PPL 比 1.385 反超裸)。
     * 补齐后 GE 与 z 的落地口径自洽: z 解的就是 GE 之后的残差。 */
    if (gecorr && Rge)
        run_map_arm("GE+4L", X, D, gecorr, tr, Rge, R, X, Ys, Yt_ev, wv, swts, ralign,
                    fit, nf, ev, mode0, nev, ranks, nrank, lambdas, nlam, maxk,
                    lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp, zcat);
    free(sw2); free(swts); free(ralign);
}

/* 数据面体检(2026-08-26 L41 爆炸定位): NaN/Inf 计数 + rms + 极值通道。
 * ds4_loss 契约"NaN-free 是调用方责任" — 深层量化前向确实会产 NaN, 这里是那道闸。 */
static void zl_healthcheck(const float *X, const float *R, const float *Ys, int ntok, int L) {
    long nx = 0, nr = 0, ny = 0; double sx = 0, sr = 0, sy = 0, mx = 0, mr = 0;
    size_t nn = (size_t)ntok * D;
    for (size_t i = 0; i < nn; i++) {
        float a = X[i], b = R[i], c = Ys[i];
        if (!isfinite(a)) nx++; else { sx += (double)a * a; if (fabsf(a) > mx) mx = fabsf(a); }
        if (!isfinite(b)) nr++; else { sr += (double)b * b; if (fabsf(b) > mr) mr = fabsf(b); }
        if (!isfinite(c)) ny++; else sy += (double)c * c;
    }
    printf("  [体检]L%d 非有限 x=%ld R=%ld Ys=%ld | rms x=%.4g R=%.4g Ys=%.4g | max|x|=%.4g max|R|=%.4g\n",
           L, nx, nr, ny, sqrt(sx / nn), sqrt(sr / nn), sqrt(sy / nn), mx, mr);
    {   /* 巨值集中度: 逐通道能量占比 top-8(L41 max/rms=1000 → 最小二乘容量被少数通道吃光) */
        double *ce = calloc(D, sizeof(double));
        if (ce) {
            for (int t = 0; t < ntok; t++)
                for (int j = 0; j < D; j++) { double v = R[(size_t)t * D + j]; ce[j] += v * v; }
            double tot = 0; for (int j = 0; j < D; j++) tot += ce[j];
            double top8 = 0; int idx8[8] = {0};
            for (int r2 = 0; r2 < 8; r2++) {
                int bi = 0; double bv = -1;
                for (int j = 0; j < D; j++) {
                    int used = 0; for (int q = 0; q < r2; q++) if (idx8[q] == j) used = 1;
                    if (!used && ce[j] > bv) { bv = ce[j]; bi = j; }
                }
                idx8[r2] = bi; top8 += bv;
            }
            printf("  [体检]L%d R 能量: top8 通道占 %.1f%% (ch %d,%d,%d…) — 巨值集中度\n",
                   L, 100.0 * top8 / (tot + 1e-30), idx8[0], idx8[1], idx8[2]);
            free(ce);
        }
    }
    if (nx + nr + ny) printf("  ★L%d 数据面含非有限值 — 解算必炸, 这是 bug 不是层的属性★\n", L);
}

/* ★fp16 往返自检(2026-08-26 用户判"引擎磨平收益"的代码级检验)★
 * 层内评估在 f32 上做, 落地 zl.RRR 是 fp16(判决尺/引擎都按 fp16 读) —— k 大时高阶
 * 奇异方向元素小, fp16 舍入可能把它们抹平甚至变噪声。本函数把 z/U/V 做一次 fp16
 * 往返(与 rec_rrr 落地位宽逐位同), 供调用方用【部署真值】重评。 */
static void z_fp16_roundtrip(ds4_z *zl) {
    for (uint32_t c = 0; c < zl->k; c++)
        zl->z[c] = ds4_f16_to_f32(ds4_f64_to_f16((double)zl->z[c]));
    for (size_t i = 0; i < (size_t)zl->d_out * zl->rank; i++)
        zl->U[i] = ds4_f16_to_f32(ds4_f64_to_f16((double)zl->U[i]));
    for (size_t i = 0; i < (size_t)zl->d_in * zl->rank; i++)
        zl->V[i] = ds4_f16_to_f32(ds4_f64_to_f16((double)zl->V[i]));
}

/* fp16 往返复评: 用【落地位宽】的 z 重算 held 四损失 —— f32 账与 fp16 账的差,
 * 就是"层内赢、端到端不兑现"里被位宽磨掉的那部分(用户 2026-08-26 判点)。 */
static void z_fp16_recheck(const best_t *best, const float *bc, const float *X,
                           const float *Ys, const float *R, const float *Yt_ev,
                           const float *wv, const int *ev, const int *mode0, int nev,
                           double dscale, uint64_t seed, int nth, int L, FILE *lf) {
    if (!best->zkeep) return;
    ds4_z *cp = z_clone_k(best->zkeep);
    z_fp16_roundtrip(cp);
    float *Yh = xmalloc((size_t)nev * D * sizeof(float));
    float *Cb = xmalloc((size_t)nev * D * sizeof(float));
    float *Cp = xmalloc((size_t)nev * D * sizeof(float));
    ds4_z *zl1[1] = { cp };
    double er = eval_apply(zl1, NULL, 1, 0, bc, 0.5f, X, Ys, R, ev, mode0, nev,
                           Yh, Cb, Cp, dscale, seed, nth);
    float la = ds4_loss_align(Yh, Yt_ev, (uint32_t)nev, D);
    float lc = ds4_loss_classify(Yh, Yt_ev, wv, (uint32_t)nev, D);
    printf("★L%d fp16复评(部署位宽): align %.4f→%.4f cls %.4f→%.4f ER %.2f%%→%.2f%% k=%u\n",
           L, best->la, la, best->lc, lc, best->er * 100, er * 100, cp->k);
    if (lf) fprintf(lf, "# fp16复评 align=%.6f cls=%.6f ER=%.2f%% (f32 账 align=%.6f ER=%.2f%%)\n",
                    la, lc, er * 100, best->la, best->er * 100);
    free(Yh); free(Cb); free(Cp);
    ds4_z_free(cp);
}
