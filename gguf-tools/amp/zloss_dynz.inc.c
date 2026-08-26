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
        p1[i] = gate2 ? tanhf(p1[i] / sv[c]) * tanhf(p2[i] / sa[c])
                      : tanhf(p1[i] / sv[c]);      /* 单门变体: 实现自检对照 */
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
        for (int gate2 = 0; gate2 <= 1; gate2++) {
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
            run_bc_arm(gate2 ? "dynz" : "dyn1", Uf2, k * D, corr, X, Ys, Yt_ev, wv,
                       R, ev, mode0, nev, lw, dscale, seed, nth, L, lf, best, Yhat, Cb, Cp);
            printf("    ↑%s k=%d λ=%.3g fitER=%.1f%% vol=%.1fMB\n", gate2 ? "dynz" : "dyn1",
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
