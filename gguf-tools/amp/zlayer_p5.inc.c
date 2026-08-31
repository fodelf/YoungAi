    /* --- 活 k_L: held 段 k 曲线 --- */
    /* ★held 打分=部署同式(2026-08-31)★ 落盘是 f16, 回放还有 ‖Δ‖≤tr·‖routed‖ 整行
     * 夹持(zreplay) —— 旧评估两样都不建模: 选 K/过闸全用 f64 无夹持数字, 层内挽回率
     * 系统性高于盘上能兑现的量("层内好端到端差"的评估侧来源)。修法两刀:
     * ①解算因子就地舍到 f16 格点(p7 emit 再舍=幂等, 载荷字节不变);
     * ②每行夹持基=学生 routed 行范数(XCAP 用引擎捕获 YQE, 否则 Σw·pYQ 重建;
     *   GE 折权对基范数是 1e-3 级且 GE 在 z 后才解, 不进基)。 */
    { double *fs[6] = {A, S, Bt, Af, Sf, Bf};
      size_t ns[6] = {(size_t)D * rlin, (size_t)rlin, (size_t)rlin * D,
                      (size_t)DIN3 * rfta, (size_t)rfta, (size_t)rfta * D};
      for (int fi = 0; fi < 6; fi++) if (fs[fi])
          for (size_t i2 = 0; i2 < ns[fi]; i2++)
              fs[fi][i2] = (double)f16_to_f32(f64_to_f16(fs[fi][i2])); }
    float *bnorm = (float *)xmalloc((size_t)NTOK * 4);
    if (XCAP && YQE) {
        for (int t2 = 0; t2 < NTOK; t2++) { double s2 = 0;
            const float *y = YQE + (size_t)t2 * D;
            for (int j = 0; j < D; j++) s2 += (double)y[j] * y[j];
            bnorm[t2] = (float)sqrt(s2); }
    } else {
        float *ys = (float *)xcalloc((size_t)NTOK * D, 4);
        for (long long p2 = 0; p2 < npair; p2++) {
            float *dst = ys + (size_t)prow[p2] * D;
            const float *y = pYQ + (size_t)p2 * D;
            for (int j = 0; j < D; j++) dst[j] += pw[p2] * y[j]; }
        for (int t2 = 0; t2 < NTOK; t2++) { double s2 = 0;
            const float *y = ys + (size_t)t2 * D;
            for (int j = 0; j < D; j++) s2 += (double)y[j] * y[j];
            bnorm[t2] = (float)sqrt(s2); }
        free(ys);
    }
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
            /* ★方向筛选(2026-08-31 用户令"找到没有肉的问题")★ SVD 前缀截断按能量序取头,
             * 深层能量头部=massive 通道舍入噪声方向 ⇒ 任何 k 都被迫先装噪声, held 全 k≈0
             * ="深层没肉"假象(L20/22 实测靶占比 32% 而 z 只收 0.5-1.8%)。解跨度不动, 按
             * held 逐方向增益(2⟨dH,p_c⟩−‖p_c‖², 独立近似)把因子列重排, 噪声方向沉底;
             * 部署=分量求和, 列序透明, 下游前缀-K 语义不变。held 复用风险由净行闸兜底。 */
            {
                double *dHev = (double *)xmalloc((size_t)nev * D * sizeof(double));
                for (int i = 0; i < nev; i++) for (int j = 0; j < D; j++)
                    dHev[(size_t)i * D + j] = dH[(size_t)ev[i] * D + j];
                double *T = (double *)xmalloc((size_t)nev * r * sizeof(double));
                mm64(0, 1, nev, r, D, dHev, D, BB, D, T, r);
                free(dHev);
                double *gain = (double *)xmalloc((size_t)r * sizeof(double));
                for (int c = 0; c < r; c++) {
                    double bb = 0, sp = 0, s22 = 0;
                    for (int j = 0; j < D; j++) bb += BB[(size_t)c * D + j] * BB[(size_t)c * D + j];
                    for (int i = 0; i < nev; i++) {
                        const double pv = Pev[(size_t)i * r + c];
                        sp += pv * T[(size_t)i * r + c]; s22 += pv * pv;
                    }
                    gain[c] = 2.0 * sp - s22 * bb;
                }
                free(T);
                int *ord = (int *)xmalloc((size_t)r * sizeof(int));
                for (int c = 0; c < r; c++) ord[c] = c;
                for (int a2 = 0; a2 < r; a2++) for (int b2 = a2 + 1; b2 < r; b2++)
                    if (gain[ord[b2]] > gain[ord[a2]]) { int t3 = ord[a2]; ord[a2] = ord[b2]; ord[b2] = t3; }
                double *Anc = pass ? Af : A, *Snc = pass ? Sf : S, *Bnc = pass ? Bf : Bt;
                double *Am = (double *)xmalloc((size_t)din * r * sizeof(double));
                double *Sm2 = (double *)xmalloc((size_t)r * sizeof(double));
                double *Bm = (double *)xmalloc((size_t)r * D * sizeof(double));
                double *Pm = (double *)xmalloc((size_t)nev * r * sizeof(double));
                for (int c = 0; c < r; c++) {
                    const int s3 = ord[c];
                    for (int i = 0; i < din; i++) Am[(size_t)i * r + c] = Anc[(size_t)i * r + s3];
                    Sm2[c] = Snc[s3];
                    memcpy(Bm + (size_t)c * D, Bnc + (size_t)s3 * D, (size_t)D * sizeof(double));
                    for (int i = 0; i < nev; i++) Pm[(size_t)i * r + c] = Pev[(size_t)i * r + s3];
                }
                memcpy(Anc, Am, (size_t)din * r * sizeof(double));
                memcpy(Snc, Sm2, (size_t)r * sizeof(double));
                memcpy(Bnc, Bm, (size_t)r * D * sizeof(double));
                memcpy(Pev, Pm, (size_t)nev * r * sizeof(double));
                free(Am); free(Sm2); free(Bm); free(Pm); free(gain); free(ord);
            }
            for (int ki = 0; ki < nKG; ki++) {
                int k = KG[ki];
                if (k > 0) mm64(0, 0, nev, D, k, Pev, r, BB, D, pbuf, D);
                double e1 = 0;
                for (int i = 0; i < nev; i++) {
                    const float *d2 = dH + (size_t)ev[i] * D;
                    const double *p = pbuf + (size_t)i * D;
                    double sc = 1.0;
                    if (k > 0) {                 /* 部署同式: zreplay 整行信任域夹持 */
                        double nd2 = 0;
                        for (int j = 0; j < D; j++) nd2 += p[j] * p[j];
                        const double cap = (double)DS4_AMP_ZL_TR * bnorm[ev[i]];
                        if (nd2 > cap * cap && nd2 > 0) sc = cap / sqrt(nd2);
                    }
                    for (int j = 0; j < D; j++) {
                        float res = d2[j] - (k > 0 ? (float)(sc * p[j]) : 0.0f);   /* py: 先降 f32 再平方 */
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
     * 关了 ftA 时 py 把 curvef 全填 0 ⇒ rz_fta=0.0。要是线性 k 曲线【全负】而 --gate
     * 比它更低(比如 -100), 选形态那一步就走成 `elif rz_fta>rz_lin: FORM="ftA"` ——
     * 而 Af/Sf/Bf 在 _FTA=0 时是 None, 下一行 Af[:,:K] 直接
     *   TypeError: 'NoneType' object is not subscriptable
     * C 这边照抄的话是空指针解引用(段错误), 症状比 py 还难查。判据与 py 完全一致, 只是
     * 换成一句能 grep 的停车话。产线用的 GATE 是 0 或 99, 撞不到这个角落。 */
    if (!strcmp(FORM, "ftA") && !(FTA && rfta > 0))
        die("assert 失败: 关了 ftA(--fta %d) 却把赢家判成 ftA —— 线性 k 曲线全负"
            "(rz_lin=%.4f) 且 --gate=%.4f 比它还低。.py 在这里会抛 "
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
        for (int t2 = 0; t2 < NTOK; t2++) {   /* GE 靶=夹持后残差(与 held 打分同式) */
            const double *pj = prj + (size_t)t2 * D;
            double nd2 = 0;
            for (int j = 0; j < D; j++) nd2 += pj[j] * pj[j];
            const double cap = (double)DS4_AMP_ZL_TR * bnorm[t2];
            const double sc = (nd2 > cap * cap && nd2 > 0) ? cap / sqrt(nd2) : 1.0;
            for (int j = 0; j < D; j++)
                R[(size_t)t2 * D + j] = dH[(size_t)t2 * D + j] - (float)(sc * pj[j]);
        }
        free(prj);
    }
    free(bnorm);

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
    /* ★闸判定量修正(2026-08-29 用户指出"没收益不应该不进入")★
     * 原 eff=USE_GE?comb:rz —— z 支微正时(USE_GE=0)闸只看 rz, 完全无视组合;
     * 而注入载荷是 z+GE【组合】: L42 rz=+0.2% 过闸, 注入的组合实测 −3.8% 有害。
     * 闸必须判【注进去的那个东西】的实测收益 = comb(GE 关时 comb=rz, 语义不变)。 */
    eff = comb;

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
        /* ★显著性闸(2026-08-29 用户裁决"没收益不进入"落到底)★ comb 微正(0.0x%)是 held
         * 噪声级, GATE=0 拦不住(L32 组合"0.0%"实为 0.0x 微正, 11.9MB 照注)。零新常数判据:
         * 分域终验两半(prog/fin)【同向为正】才算真收益 —— 真实层(≥0.8%)两半同正, 噪声层
         * 一正一负。不同向 ⇒ eff 记 0, p7 的闸自动拒。NAN(半区空)同拒, 保守方向。 */
        if (eff > 0 && !(dv[0] > 0 && dv[1] > 0)) eff = 0.0;
    }
    /* ★跨语料闸(--gate-anchor, 2026-08-31)★ 三等分语料后闸一直只看同语料 held ⇒
     * 256 门+rank64 图对拟合语料过拟合照样过闸(同文档相邻段同分布, held 虚高)。
     * 闸料=量化半锚: 与反修半零重叠、又不是判决锚(判决锚进闸=对判决做模型选择,
     * run_p4 路由 FIT 同款铁律禁止)。口径与主解算同(FP 自洽, 同 dql 学生), 打分与
     * held 同式(f16 因子+整行夹持+GE)。行=全锚等距抽(闸锚行序是域连续块, 取头 N 行
     * 只会闸到第一个域); 闸是读数不是拟合, 默认 2048 行足够判符号。挽回≤0 ⇒ 拒注。
     * ADDON 不闸(部署 op 是新旧合并体, 单评新解=错对象; 冠军配方非 ADDON)。 */
    if (GATEA && !ADDON && eff > GATE && (K > 0 || USE_GE)) {
        int Sg = 0;
        { FILE *gfp = fopen(GATEA, "rb"); uint32_t h8[8];
          if (!gfp || fread(h8, 4, 8, gfp) != 8) die("闸锚打不开: %s", GATEA);
          fclose(gfp); Sg = (int)h8[1]; }
        float *Xg = NULL, *rwg = NULL; int *ridxg = NULL; ameta_t amg;
        anchor_layer(GATEA, L, Sg, &Xg, &ridxg, &rwg, &amg);
        if (amg.NACT != am.NACT) die("闸锚 NACT=%d ≠ 主锚 %d — 口径不明拒闸", amg.NACT, am.NACT);
        /* ★闸行必须过行掩码(2026-08-31 探针实锤)★ 旧版盲等距抽全锚, 吃进拼接毒行
         * (每 128 行窗头 32 行=全锚 25%, 语料切窗的上下文断点)。毒行活性浅层正常、
         * 深层滚雪球: L31 实测闸靶RMS 5.9e-1 = held 侧(掩码后) 1.5e-1 的 4×, 基RMS
         * 11×于 L20 —— 深层闸读数被毒行统治(−149~−280% 与 +89/+97% 同带并存), 拒注
         * 判的是毒不是过拟合。净行=闸锚自己 .layout 的 fit∪eval(两表各自升序, 归并),
         * 再等距。归并后 cln[i]≥i ⇒ 压紧仍 src≥dst 前向安全。 */
        int *gtr = NULL, *gev = NULL; int gntr = 0, gnev = 0;
        if (row_layout_split(GATEA, &gtr, &gntr, &gev, &gnev) != 0)
            die("闸锚行布局缺 %s.layout — 先跑 amp_campaign.sh anchors3 补", GATEA);
        int ncln = gntr + gnev;
        int *cln = (int *)xmalloc((size_t)ncln * sizeof(int));
        { int a2 = 0, b2 = 0, w2 = 0, mono = 1;
          while (a2 < gntr || b2 < gnev)
              cln[w2++] = (b2 >= gnev || (a2 < gntr && gtr[a2] <= gev[b2])) ? gtr[a2++] : gev[b2++];
          for (int i = 1; i < ncln; i++) if (cln[i] <= cln[i - 1]) mono = 0;
          if (!mono || cln[ncln - 1] >= Sg) die("闸锚布局行序异常 — 停车不带毒判"); }
        free(gtr); free(gev);
        int NG = NGATE > ncln ? ncln : NGATE;
        const int stepc = ncln / NG;
        for (int i = 0; i < NG; i++) {          /* 净行等距压紧到前 NG 行 */
            int s2 = cln[(size_t)i * stepc];
            memmove(Xg + (size_t)i * D, Xg + (size_t)s2 * D, (size_t)D * 4);
            memmove(ridxg + (size_t)i * amg.NACT, ridxg + (size_t)s2 * amg.NACT, (size_t)amg.NACT * 4);
            memmove(rwg + (size_t)i * amg.NACT, rwg + (size_t)s2 * amg.NACT, (size_t)amg.NACT * 4);
        }
        free(cln);
        float *dHg, *pwg, *pYQg; int *prowg, *peg; long long npg;
        zl_build_pairs(hf, ld, GGP, &gg, L, NG, amg.NACT, SWLIM, Xg, ridxg, rwg,
                       0, NULL, NULL, NULL, NULL, 1, &dHg, &prowg, &peg, &pwg, &pYQg, &npg);
        float *ysg = (float *)xcalloc((size_t)NG * D, 4);   /* 学生基行范数=夹持 cap */
        for (long long p2 = 0; p2 < npg; p2++) {
            float *dst = ysg + (size_t)prowg[p2] * D;
            const float *y = pYQg + (size_t)p2 * D;
            for (int j = 0; j < D; j++) dst[j] += pwg[p2] * y[j];
        }
        float *corr = (float *)xcalloc((size_t)NG * D, 4);
        if (K > 0) {                            /* 冠军形态投影(因子已在 f16 格点) */
            const int ftaW = !strcmp(FORM, "ftA");
            const int din = ftaW ? DIN3 : D, r = ftaW ? rfta : rlin;
            const double *AA = ftaW ? Af : A, *SS2 = ftaW ? Sf : S, *BB = ftaW ? Bf : Bt;
            double *Ph = (double *)xmalloc((size_t)NG * din * sizeof(double));
            if (ftaW) {
                float *P32 = (float *)xmalloc((size_t)NG * DIN3 * 4);
                phi32_ctx pc = {Xg, P32, D};
                parallel_for(NG, phi32_worker, &pc);
                for (size_t i2 = 0; i2 < (size_t)NG * DIN3; i2++) Ph[i2] = P32[i2];
                free(P32);
            } else for (size_t i2 = 0; i2 < (size_t)NG * D; i2++) Ph[i2] = Xg[i2];
            double *AS = (double *)xmalloc((size_t)din * K * sizeof(double));
            for (int i2 = 0; i2 < din; i2++) for (int c = 0; c < K; c++)
                AS[(size_t)i2 * K + c] = AA[(size_t)i2 * r + c] * SS2[c];
            double *Pv = (double *)xmalloc((size_t)NG * K * sizeof(double));
            mm64(0, 0, NG, K, din, Ph, din, AS, K, Pv, K);
            double *pr2 = (double *)xmalloc((size_t)NG * D * sizeof(double));
            mm64(0, 0, NG, D, K, Pv, K, BB, D, pr2, D);
            for (int t2 = 0; t2 < NG; t2++) {   /* 部署同式整行夹持 */
                double nd2 = 0, nb2 = 0;
                for (int j = 0; j < D; j++) {
                    nd2 += pr2[(size_t)t2 * D + j] * pr2[(size_t)t2 * D + j];
                    nb2 += (double)ysg[(size_t)t2 * D + j] * ysg[(size_t)t2 * D + j];
                }
                const double cap = (double)DS4_AMP_ZL_TR * sqrt(nb2);
                const double sc2 = (nd2 > cap * cap && nd2 > 0) ? cap / sqrt(nd2) : 1.0;
                for (int j = 0; j < D; j++) corr[(size_t)t2 * D + j] = (float)(sc2 * pr2[(size_t)t2 * D + j]);
            }
            free(Ph); free(AS); free(Pv); free(pr2);
        }
        if (USE_GE) for (long long p2 = 0; p2 < npg; p2++) {
            const float g1 = gf[peg[p2]] - 1.0f;
            float *dst = corr + (size_t)prowg[p2] * D;
            const float *y = pYQg + (size_t)p2 * D;
            for (int j = 0; j < D; j++) dst[j] += g1 * pwg[p2] * y[j];
        }
        double eg0 = 0, eg1 = 0;
        for (size_t i2 = 0; i2 < (size_t)NG * D; i2++) {
            const double e2 = (double)dHg[i2] - corr[i2];
            eg1 += e2 * e2; eg0 += (double)dHg[i2] * dHg[i2];
        }
        const double recg = eg0 > 0 ? 1.0 - eg1 / eg0 : 0.0;
        /* 靶量级随行打印(2026-08-31): 挽回是比值, FP 靶塌缩层分母≈0 ⇒ 读数±爆表
         * (champ3 实况: L31-L37 −91~−280% 与 L35/38 +89/+97% 同深度带并存)。
         * 靶占比 = 靶RMS/学生基RMS = 该层量化误差在 FP-x 口径下的相对大小 —— 没有它,
         * "挽回 -225%"与"挽回 97%"分不清是信号还是分母噪声。 */
        double gb = 0;
        for (size_t i2 = 0; i2 < (size_t)NG * D; i2++) gb += (double)ysg[i2] * ysg[i2];
        printf("  L%d 跨语料闸(量化半 %d 行): 挽回 %.2f%% [靶RMS %.3e 基RMS %.3e 靶占比 %.2f%% | held靶RMS %.3e] → %s\n",
               L, NG, recg * 100, sqrt(eg0 / ((double)NG * D)), sqrt(gb / ((double)NG * D)),
               gb > 0 ? 100.0 * sqrt(eg0 / gb) : -1.0, nev > 0 ? sqrt(e0 / ((double)nev * D)) : 0.0,
               recg > 0 ? "通过" : "★拒: 同语料 held 正/跨语料负 = 过拟合★");
        if (recg <= 0) eff = 0.0;
        free(Xg); free(ridxg); free(rwg); free(dHg); free(prowg); free(peg);
        free(pwg); free(pYQg); free(ysg); free(corr);
    }
    double t2 = now_s();

