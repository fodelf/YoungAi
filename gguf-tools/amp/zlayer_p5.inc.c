    /* --- 活 k_L: held 段 k 曲线 --- */
    double e0 = 0;
    for (int i = 0; i < nev; i++) { const float *d2 = dH + (size_t)ev[i] * D;
        for (int j = 0; j < D; j++) e0 += (double)d2[j] * d2[j]; }

    double curve[16], curvef[16];
    for (int i = 0; i < nKG; i++) { curve[i] = 0.0; curvef[i] = 0.0; }
    {
        double *pbuf = (double *)xmalloc((size_t)nev * D * sizeof(double));
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1 && !(FTA && rfta > 0)) break;
            int r = pass ? rfta : rlin, din = pass ? DIN3 : D;
            const double *AA = pass ? Af : A, *SS = pass ? Sf : S, *BB = pass ? Bf : Bt;
            if (r <= 0) continue;
            double *Ph = (double *)xmalloc((size_t)nev * din * sizeof(double));
            if (pass) { for (size_t i = 0; i < (size_t)nev * din; i++) Ph[i] = Phi_ev32[i]; }
            else { for (int i = 0; i < nev; i++) memcpy(Ph + (size_t)i * D, X + (size_t)ev[i] * D, (size_t)D * sizeof(double)); }
            double *AS = (double *)xmalloc((size_t)din * r * sizeof(double));
            for (int i = 0; i < din; i++) for (int c = 0; c < r; c++) AS[(size_t)i * r + c] = AA[(size_t)i * r + c] * SS[c];
            double *Pev = (double *)xmalloc((size_t)nev * r * sizeof(double));
            mm64(0, 0, nev, r, din, Ph, din, AS, r, Pev, r);
            free(Ph); free(AS);
            for (int ki = 0; ki < nKG; ki++) {
                int k = KG[ki];
                if (k > 0) mm64(0, 0, nev, D, k, Pev, r, BB, D, pbuf, D);
                double e1 = 0;
                for (int i = 0; i < nev; i++) {
                    const float *d2 = dH + (size_t)ev[i] * D;
                    const double *p = pbuf + (size_t)i * D;
                    for (int j = 0; j < D; j++) {
                        float res = d2[j] - (k > 0 ? (float)p[j] : 0.0f);   /* py: 先降 f32 再平方 */
                        e1 += (double)res * res;
                    }
                }
                double v = 1.0 - e1 / e0;
                if (pass) curvef[ki] = v; else curve[ki] = v;
            }
            free(Pev);
        }
        free(pbuf);
    }

    double rz_lin = -1e300, rz_fta = -1e300;
    for (int i = 0; i < nKG; i++) { if (curve[i] > rz_lin) rz_lin = curve[i]; if (curvef[i] > rz_fta) rz_fta = curvef[i]; }
    if (!FTA) { rz_fta = 0.0; for (int i = 0; i < nKG; i++) curvef[i] = 0.0; }
    const char *FORM = "lin";
    double rz;
    if ((rz_lin > rz_fta ? rz_lin : rz_fta) <= GATE) { K = 0; rz = 0.0; }
    else if (rz_fta > rz_lin) {
        FORM = "ftA"; K = KG[0]; rz = curvef[0];
        for (int i = 1; i < nKG; i++) if (curvef[i] > rz) { rz = curvef[i]; K = KG[i]; }
    } else {
        K = KG[0]; rz = curve[0];
        for (int i = 1; i < nKG; i++) if (curve[i] > rz) { rz = curve[i]; K = KG[i]; }
    }
    printf("  L%d k曲线lin", L);
    for (int i = 0; i < nKG; i++) printf("%s%d:%.1f", i ? " " : " ", KG[i], curve[i] * 100);
    printf("\n");
    if (FTA) {
        printf("  L%d k曲线ftA", L);
        for (int i = 0; i < nKG; i++) printf("%s%d:%.1f", i ? " " : " ", KG[i], curvef[i] * 100);
        printf(" → 赢家=%s k_L=%d\n", FORM, K);
    } else printf("  L%d 纯z(ftA关) → k_L=%d\n", L, K);

    /* R = dH − 中标形态的 z 出力(f32, 与 py 的 .astype(np.float32) 同位置) */
    /* ★.py 的一个潜伏坑, 这里改成停车而不是跟着崩★
     * 关了 ftA 时 py 把 curvef 全填 0 ⇒ rz_fta=0.0。要是线性 k 曲线【全负】而 DS4_ZL_GATE
     * 比它更低(比如 -100), 选形态那一步就走成 `elif rz_fta>rz_lin: FORM="ftA"` ——
     * 而 Af/Sf/Bf 在 _FTA=0 时是 None, 下一行 Af[:,:K] 直接
     *   TypeError: 'NoneType' object is not subscriptable
     * C 这边照抄的话是空指针解引用(段错误), 症状比 py 还难查。判据与 py 完全一致, 只是
     * 换成一句能 grep 的停车话。产线用的 GATE 是 0 或 99, 撞不到这个角落。 */
    if (!strcmp(FORM, "ftA") && !(FTA && rfta > 0))
        die("assert 失败: 关了 ftA(DS4_ZL_FTA=%d) 却把赢家判成 ftA —— 线性 k 曲线全负"
            "(rz_lin=%.4f) 且 DS4_ZL_GATE=%.4f 比它还低。.py 在这里会抛 "
            "TypeError: 'NoneType' object is not subscriptable。把闸调回 ≥0 即可绕开。",
            FTA, rz_lin, GATE);

    float *R = dH;
    if (K > 0) {
        int din = !strcmp(FORM, "ftA") ? DIN3 : D;
        int r = !strcmp(FORM, "ftA") ? rfta : rlin;
        const double *AA = !strcmp(FORM, "ftA") ? Af : A, *SS = !strcmp(FORM, "ftA") ? Sf : S,
                     *BB = !strcmp(FORM, "ftA") ? Bf : Bt;
        double *Ph = (double *)xmalloc((size_t)NTOK * din * sizeof(double));
        if (din == DIN3) {
            float *X32 = (float *)xmalloc((size_t)NTOK * D * 4);
            for (size_t i = 0; i < (size_t)NTOK * D; i++) X32[i] = (float)X[i];
            float *P32 = (float *)xmalloc((size_t)NTOK * DIN3 * 4);
            { phi32_ctx pc = {X32, P32, D}; parallel_for(NTOK, phi32_worker, &pc); }
            for (size_t i = 0; i < (size_t)NTOK * DIN3; i++) Ph[i] = P32[i];
            free(X32); free(P32);
        } else memcpy(Ph, X, (size_t)NTOK * D * sizeof(double));
        double *AS = (double *)xmalloc((size_t)din * K * sizeof(double));
        for (int i = 0; i < din; i++) for (int c = 0; c < K; c++) AS[(size_t)i * K + c] = AA[(size_t)i * r + c] * SS[c];
        double *Pv = (double *)xmalloc((size_t)NTOK * K * sizeof(double));
        mm64(0, 0, NTOK, K, din, Ph, din, AS, K, Pv, K);
        free(Ph); free(AS);
        double *prj = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
        mm64(0, 0, NTOK, D, K, Pv, K, BB, D, prj, D);
        free(Pv);
        R = (float *)xmalloc((size_t)NTOK * D * 4);
        for (size_t i = 0; i < (size_t)NTOK * D; i++) R[i] = dH[i] - (float)prj[i];
        free(prj);
    }

    /* tok_pairs: 每 token 的配对下标(CSR), 与 py 的 append 顺序一致(p 升序) */
    int *pcnt = (int *)xcalloc((size_t)NTOK + 1, sizeof(int));
    for (long long p = 0; p < npair; p++) pcnt[prow[p] + 1]++;
    for (int t = 0; t < NTOK; t++) pcnt[t + 1] += pcnt[t];
    int *pidx = (int *)xmalloc((size_t)(npair ? npair : 1) * sizeof(int));
    { int *fill = (int *)xmalloc(((size_t)NTOK + 1) * sizeof(int));
      memcpy(fill, pcnt, ((size_t)NTOK + 1) * sizeof(int));
      for (long long p = 0; p < npair; p++) pidx[fill[prow[p]]++] = (int)p;
      free(fill); }
    int maxpp = 0;
    for (int t = 0; t < NTOK; t++) if (pcnt[t + 1] - pcnt[t] > maxpp) maxpp = pcnt[t + 1] - pcnt[t];

    /* ---------------- GE: 每专家门(k 截断后残差 ridge) ---------------- */
    uint16_t ge16[NEXP];
    float gf[NEXP];
    double comb, eff;
    int USE_GE = 0;
    if (GE_ON) {
        double *Gg = (double *)xcalloc(NEXP * NEXP, sizeof(double));
        double *bg = (double *)xcalloc(NEXP, sizeof(double));
        double *va = (double *)xmalloc((size_t)(maxpp ? maxpp : 1) * D * sizeof(double));
        int *ve = (int *)xmalloc((size_t)(maxpp ? maxpp : 1) * sizeof(int));
        for (int i = 0; i < ntr; i++) {
            int t = tr[i], n = pcnt[t + 1] - pcnt[t];
            for (int a = 0; a < n; a++) {
                int p = pidx[pcnt[t] + a];
                ve[a] = pe[p];
                const float *y = pYQ + (size_t)p * D;
                double *v = va + (size_t)a * D;
                float w = pw[p];
                for (int j = 0; j < D; j++) v[j] = (double)(w * y[j]);   /* py: f32 乘积再升 f64 */
            }
            const float *rt = R + (size_t)t * D;
            for (int a = 0; a < n; a++) {
                const double *v = va + (size_t)a * D;
                double s = 0;
                for (int j = 0; j < D; j++) s += v[j] * rt[j];
                bg[ve[a]] += s;
                for (int b = a; b < n; b++) {
                    const double *vb = va + (size_t)b * D;
                    double d2 = 0;
                    for (int j = 0; j < D; j++) d2 += v[j] * vb[j];
                    Gg[(size_t)ve[a] * NEXP + ve[b]] += d2;
                    if (ve[a] != ve[b]) Gg[(size_t)ve[b] * NEXP + ve[a]] += d2;
                }
            }
        }
        free(va); free(ve);
        double gtr = 0;
        for (int i = 0; i < NEXP; i++) gtr += Gg[(size_t)i * NEXP + i];
        double shr = GELAM * (gtr / NEXP > 1.0 ? gtr / NEXP : 1.0);
        for (int i = 0; i < NEXP; i++) Gg[(size_t)i * NEXP + i] += shr;
        if (lu_solve(Gg, NEXP, bg)) die("GE 解算奇异 — 停车");
        for (int e = 0; e < NEXP; e++) { ge16[e] = f64_to_f16(1.0 + bg[e]); gf[e] = f16_to_f32(ge16[e]); }
        free(Gg); free(bg);
        /* 组合终验: z^L(K)+GE 在 held 段的总增益 */
        double eng = 0;
        double *rr = (double *)xmalloc((size_t)D * sizeof(double));
        for (int i = 0; i < nev; i++) {
            int t = ev[i];
            const float *rt = R + (size_t)t * D;
            for (int j = 0; j < D; j++) rr[j] = rt[j];
            for (int a = pcnt[t]; a < pcnt[t + 1]; a++) {
                int p = pidx[a];
                double g1 = (double)gf[pe[p]] - 1.0;
                const float *y = pYQ + (size_t)p * D;
                float w = pw[p];
                for (int j = 0; j < D; j++) rr[j] -= g1 * (double)(w * y[j]);
            }
            for (int j = 0; j < D; j++) eng += rr[j] * rr[j];
        }
        free(rr);
        comb = 1.0 - eng / e0;
        USE_GE = (comb >= rz - 1e-9) || (K == 0);
    } else {
        for (int e = 0; e < NEXP; e++) { ge16[e] = f64_to_f16(1.0); gf[e] = 1.0f; }
        comb = rz; USE_GE = 0;
    }
    eff = USE_GE ? comb : rz;

    /* 分域终验: held 前半=prog 后半=fin(注意 py 这里【总是】叠 GE 效应, GE 关时 gf≡1 无害) */
    if (nev >= 2) {
        int half = nev / 2;
        double dv[2];
        for (int side = 0; side < 2; side++) {
            const int *idx = side ? ev + half : ev;
            int n = side ? nev - half : half;
            if (n == 0) { dv[side] = NAN; continue; }
            double e0d = 0;
            for (int i = 0; i < n; i++) { const float *d2 = dH + (size_t)idx[i] * D;
                for (int j = 0; j < D; j++) e0d += (double)d2[j] * d2[j]; }
            if (e0d <= 0) { dv[side] = NAN; continue; }
            double ed = 0;
            double *rr = (double *)xmalloc((size_t)D * sizeof(double));
            for (int i = 0; i < n; i++) {
                int t = idx[i];
                const float *rt = (K == 0 ? dH : R) + (size_t)t * D;
                for (int j = 0; j < D; j++) rr[j] = rt[j];
                for (int a = pcnt[t]; a < pcnt[t + 1]; a++) {
                    int p = pidx[a];
                    double g1 = (double)gf[pe[p]] - 1.0;
                    const float *y = pYQ + (size_t)p * D;
                    float w = pw[p];
                    for (int j = 0; j < D; j++) rr[j] -= g1 * (double)(w * y[j]);
                }
                for (int j = 0; j < D; j++) ed += rr[j] * rr[j];
            }
            free(rr);
            dv[side] = 1.0 - ed / e0d;
        }
        printf("  L%d 分域组合: prog=%.1f%%  fin=%.1f%%  [组件门: z=%.1f z+GE=%.1f → %s]\n",
               L, dv[0] * 100, dv[1] * 100, rz * 100, comb * 100,
               (USE_GE && K > 0) ? "z+GE" : (K == 0 ? "GE-only" : "纯z"));
    }
    double t2 = now_s();

