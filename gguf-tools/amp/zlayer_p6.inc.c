    /* ================= ERF 死层部件 =================
     * 触发线(py): (not ADDON) and eff < DS4_ZL_ERF_BAR(0.01) and DS4_ZL_ERF。
     * 做的事: 逐专家在【已中标 z 与 GE 之上】的真残差里, 用 ΔW_w2 = W_fp − W_q 的
     * 加权低秩方向再修一刀 ——
     *   ① 用该专家在 fit 行上的隐层能量 _sh 给 ΔW_w2 的列加权, 取 r 个主方向(randomized SVD);
     *   ② 每个方向一个系数 α(小 r×r 岭回归), 得到每个 token 的修正量 _ct;
     *   ③ token 能量门: 按 _ct 能量降序累积收益, 取收益最大的截断点 τ, 能量 < τ 的 token 不修;
     *   ④ 中标了就把 _ct 从累积残差里扣掉 —— 后一个专家看到的是前面都修完的残差。
     * 记录 zl.ERF 载荷 = <IHH>(ne, r, 0) + 逐专家 <If>(e, τ) + U16[D,r] + V16[r,F]。
     * ★不逐位★ randomized SVD 里有 QR(见 hh_qr 注释)与一次小 SVD; U/V 载荷不与 .py 逐字节同。
     * 逐位的是: 专家序、专家数 ne、记录头、记录总长度; τ 与挽回率对到打印精度。 */
    uint8_t *ERF_ADD = NULL; size_t ERF_LEN = 0;
    double ERF_GAIN = 0.0; int ERF_NE = 0;
    if (!ADDON && eff < ERF_BAR && ERF_EN) {
        snprintf(path, sizeof path, "%s/dql_vq_L%02d.bin", ld, L);
        int efd = open(path, O_RDONLY);
        if (efd < 0) {
            /* py 这里整块套在 try/except 里: GGUF 底座根本没有 dql_vq, 打不开就是这条路径。
             * 照抄它的行为(打一行 异常 就继续), 不停车。 */
            printf("  L%d ERF死层部件异常: 打不开 %s(%s)\n", L, path, strerror(errno));
        } else {
            struct stat ebst; fstat(efd, &ebst);
            size_t ebsz = (size_t)ebst.st_size;
            const uint8_t *eblob = (const uint8_t *)mmap(NULL, ebsz, PROT_READ, MAP_PRIVATE, efd, 0);
            if (eblob == MAP_FAILED) die("ERF: VQ 侧车 mmap 失败: %s", path);
            close(efd);
            st_ctx *esc = (st_ctx *)xmalloc(sizeof(st_ctx));
            st_open(esc, hf);
            const int ERF_R = ERF_RANK;
            if (ERF_R < 1 || ERF_R > 256) die("--erf-r %d 超范围", ERF_R);
            const int RP8 = ERF_R + 8;                       /* py: _rsvd 的过采样 k+8 */

            char *trm = (char *)xcalloc((size_t)NTOK, 1), *evm = (char *)xcalloc((size_t)NTOK, 1);
            for (int i = 0; i < ntr; i++) trm[tr[i]] = 1;
            for (int i = 0; i < nev; i++) evm[ev[i]] = 1;
            int *evpos = (int *)xmalloc((size_t)NTOK * sizeof(int));
            for (int t = 0; t < NTOK; t++) evpos[t] = -1;
            for (int i = 0; i < nev; i++) evpos[ev[i]] = i;

            /* 叠加基残差: _R = (K==0 ? dH : R) 再扣掉 GE 增益效应(USE_GE 时) */
            double *ER = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
            { const float *src = (K == 0) ? dH : R;
              for (size_t i = 0; i < (size_t)NTOK * D; i++) ER[i] = src[i]; }
            if (USE_GE) {
                for (int t = 0; t < NTOK; t++) {
                    double *r0 = ER + (size_t)t * D;
                    for (int a = pcnt[t]; a < pcnt[t + 1]; a++) {
                        int p = pidx[a];
                        double g1 = (double)gf[pe[p]] - 1.0;
                        const float *y = pYQ + (size_t)p * D;
                        float w = pw[p];
                        for (int j = 0; j < D; j++) r0[j] -= g1 * (double)(w * y[j]);
                    }
                }
            }
            double *Rev0 = (double *)xmalloc((size_t)nev * D * sizeof(double));
            for (int i = 0; i < nev; i++) memcpy(Rev0 + (size_t)i * D, ER + (size_t)ev[i] * D, (size_t)D * sizeof(double));
            double *pred = (double *)xcalloc((size_t)nev * D, sizeof(double));

            /* _order = 按该专家的配对数降序(py 的 sorted 是稳定的 → 同数按专家号升序), 取前 128 */
            int ecnt[NEXP]; memset(ecnt, 0, sizeof ecnt);
            for (long long p = 0; p < npair; p++) ecnt[pe[p]]++;
            dsort_t eorder[NEXP]; int neo = 0;
            for (int e = 0; e < NEXP; e++) if (ecnt[e] > 0) { eorder[neo].v = (double)ecnt[e]; eorder[neo].i = e; neo++; }
            qsort(eorder, (size_t)neo, sizeof(dsort_t), cmp_dsort_desc);
            if (neo > 128) neo = 128;

            /* 逐专家的配对下标(p 升序 = np.where(pe==e)[0] 的顺序) */
            int *ecsr = (int *)xcalloc((size_t)NEXP + 1, sizeof(int));
            for (long long p = 0; p < npair; p++) ecsr[pe[p] + 1]++;
            for (int e = 0; e < NEXP; e++) ecsr[e + 1] += ecsr[e];
            int *eidx = (int *)xmalloc((size_t)(npair ? npair : 1) * sizeof(int));
            { int *fill = (int *)xmalloc(((size_t)NEXP + 1) * sizeof(int));
              memcpy(fill, ecsr, ((size_t)NEXP + 1) * sizeof(int));
              for (long long p = 0; p < npair; p++) eidx[fill[pe[p]]++] = (int)p;
              free(fill); }

            uint8_t *epay = NULL; size_t epay_len = 0; int ne = 0;
            int *ftr = (int *)xmalloc((size_t)(npair ? npair : 1) * sizeof(int));
            int *fev = (int *)xmalloc((size_t)(npair ? npair : 1) * sizeof(int));

            for (int oi = 0; oi < neo; oi++) {
                int e = eorder[oi].i;
                int nft = 0, nfe = 0;
                for (int a = ecsr[e]; a < ecsr[e + 1]; a++) {
                    int p = eidx[a];
                    if (trm[prow[p]]) ftr[nft++] = p;
                    if (evm[prow[p]]) fev[nfe++] = p;
                }
                if (nft < 24 || nfe < 2) continue;
                uint64_t o1 = vq_slot(eblob, ebsz, e, 0), o3 = vq_slot(eblob, ebsz, e, 1), o2 = vq_slot(eblob, ebsz, e, 2);
                if (!o1 || !o3 || !o2) continue;

                char tn[256];
                snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.w2.weight", L, e);
                long Rf = 0, Cf = 0;
                float *wf2 = st_read_weight(esc, tn, &Rf, &Cf);
                if (!wf2) die("ERF: HF 权重读不到 %s", tn);
                long Rq = 0, Cq = 0;
                float *wq2 = vq_dequant(eblob, ebsz, o2, &Rq, &Cq);
                if (!wq2) die("ERF: w2 VQ 载荷读不到(e=%d)", e);
                if (Rf != Rq || Cf != Cq) {                  /* py: _Wf.shape != _Wq.shape → _Wf=_Wf.T */
                    float *trm2 = (float *)xmalloc((size_t)Rf * Cf * 4);
                    for (long r2 = 0; r2 < Rf; r2++) for (long c2 = 0; c2 < Cf; c2++)
                        trm2[(size_t)c2 * Rf + r2] = wf2[(size_t)r2 * Cf + c2];
                    free(wf2); wf2 = trm2; long t2s = Rf; Rf = Cf; Cf = t2s;
                }
                if (Rf != D) die("ERF: w2 行数 %ld ≠ D=%d(e=%d)", Rf, D, e);
                const int F = (int)Cf;                        /* = MOEI */
                long R1 = 0, C1 = 0, R3 = 0, C3 = 0;
                float *w1q = vq_dequant(eblob, ebsz, o1, &R1, &C1);
                float *w3q = vq_dequant(eblob, ebsz, o3, &R3, &C3);
                if (!w1q || !w3q) die("ERF: w1/w3 VQ 载荷读不到(e=%d)", e);
                if (C1 != D) { float *t3 = (float *)xmalloc((size_t)R1 * C1 * 4);
                    for (long r2 = 0; r2 < R1; r2++) for (long c2 = 0; c2 < C1; c2++) t3[(size_t)c2 * R1 + r2] = w1q[(size_t)r2 * C1 + c2];
                    free(w1q); w1q = t3; long s = R1; R1 = C1; C1 = s; }
                if (C3 != D) { float *t3 = (float *)xmalloc((size_t)R3 * C3 * 4);
                    for (long r2 = 0; r2 < R3; r2++) for (long c2 = 0; c2 < C3; c2++) t3[(size_t)c2 * R3 + r2] = w3q[(size_t)r2 * C3 + c2];
                    free(w3q); w3q = t3; long s = R3; R3 = C3; C3 = s; }
                if (R1 != F || R3 != F || C1 != D || C3 != D)
                    die("ERF: w1/w3 形状 %ldx%ld / %ldx%ld ≠ %dx%d(e=%d)", R1, C1, R3, C3, F, D, e);

                /* _ht/_he = swiglu(x@W1qᵀ, x@W3qᵀ), 全 f64(py: _xt 是 f64, W 被 numpy 提升) */
                double *W1d = (double *)xmalloc((size_t)F * D * sizeof(double));
                double *W3d = (double *)xmalloc((size_t)F * D * sizeof(double));
                for (size_t i2 = 0; i2 < (size_t)F * D; i2++) { W1d[i2] = w1q[i2]; W3d[i2] = w3q[i2]; }
                free(w1q); free(w3q);
                double *xt = (double *)xmalloc((size_t)nft * D * sizeof(double));
                double *xe = (double *)xmalloc((size_t)nfe * D * sizeof(double));
                for (int j = 0; j < nft; j++) { const float *s = XSOLVE + (size_t)prow[ftr[j]] * D;
                    for (int c = 0; c < D; c++) xt[(size_t)j * D + c] = s[c]; }
                for (int j = 0; j < nfe; j++) { const float *s = XSOLVE + (size_t)prow[fev[j]] * D;
                    for (int c = 0; c < D; c++) xe[(size_t)j * D + c] = s[c]; }
                double *ht = (double *)xmalloc((size_t)nft * F * sizeof(double));
                double *he = (double *)xmalloc((size_t)nfe * F * sizeof(double));
                double *ubuf = (double *)xmalloc((size_t)(nft > nfe ? nft : nfe) * F * sizeof(double));
                mm64(0, 1, nft, F, D, xt, D, W1d, D, ht, F);
                mm64(0, 1, nft, F, D, xt, D, W3d, D, ubuf, F);
                for (size_t i2 = 0; i2 < (size_t)nft * F; i2++) {
                    double g = ht[i2], u = ubuf[i2];
                    if (SWLIM > 0) { if (g > SWLIM) g = SWLIM; else if (g < -SWLIM) g = -SWLIM;
                                     if (u > SWLIM) u = SWLIM; else if (u < -SWLIM) u = -SWLIM; }
                    double gc = g > 60.0 ? 60.0 : (g < -60.0 ? -60.0 : g);
                    ht[i2] = (g / (1.0 + exp(-gc))) * u;
                }
                mm64(0, 1, nfe, F, D, xe, D, W1d, D, he, F);
                mm64(0, 1, nfe, F, D, xe, D, W3d, D, ubuf, F);
                for (size_t i2 = 0; i2 < (size_t)nfe * F; i2++) {
                    double g = he[i2], u = ubuf[i2];
                    if (SWLIM > 0) { if (g > SWLIM) g = SWLIM; else if (g < -SWLIM) g = -SWLIM;
                                     if (u > SWLIM) u = SWLIM; else if (u < -SWLIM) u = -SWLIM; }
                    double gc = g > 60.0 ? 60.0 : (g < -60.0 ? -60.0 : g);
                    he[i2] = (g / (1.0 + exp(-gc))) * u;
                }
                free(xt); free(xe); free(W1d); free(W3d); free(ubuf);

                double *sh = (double *)xmalloc((size_t)F * sizeof(double));
                for (int c = 0; c < F; c++) {
                    double ss = 0;
                    for (int j = 0; j < nft; j++) { double v = ht[(size_t)j * F + c]; ss += v * v; }
                    sh[c] = sqrt(ss / nft) + 1e-8;
                }
                /* Ms = (ΔW_w2 ⊙ sh).astype(f32); 差在 f64 里算(与 py 的 _Wf/_Wq 已升 f64 同) */
                float *Ms = (float *)xmalloc((size_t)D * F * 4);
                for (int r2 = 0; r2 < D; r2++) for (int c = 0; c < F; c++)
                    Ms[(size_t)r2 * F + c] = (float)((((double)wf2[(size_t)r2 * F + c]) - (double)wq2[(size_t)r2 * F + c]) * sh[c]);
                free(wf2); free(wq2);

                /* ---- _rsvd(Ms, ERF_R): G=RandomState(11).randn(F, r+8) → QR(Ms@G) → svd(Qᵀ Ms) ---- */
                float *Gr = (float *)xmalloc((size_t)F * RP8 * 4);
                { mt_t rs; mt_seed(&rs, 11);
                  for (size_t i2 = 0; i2 < (size_t)F * RP8; i2++) Gr[i2] = (float)mt_gauss(&rs); }
                float *MG = (float *)xmalloc((size_t)D * RP8 * 4);
                mm32(0, 0, D, RP8, F, Ms, F, Gr, RP8, MG, RP8);
                free(Gr);
                double *Qd = (double *)xmalloc((size_t)D * RP8 * sizeof(double));
                for (size_t i2 = 0; i2 < (size_t)D * RP8; i2++) Qd[i2] = MG[i2];
                free(MG);
                double *qtau = (double *)xmalloc((size_t)RP8 * sizeof(double));
                double *qrd = (double *)xmalloc((size_t)RP8 * sizeof(double));
                hh_qr(Qd, D, RP8, qtau, qrd);
                double *Qm = (double *)xcalloc((size_t)D * RP8, sizeof(double));
                for (int c = 0; c < RP8; c++) Qm[(size_t)c * RP8 + c] = 1.0;   /* Q = Q_full·I[:, :RP8] */
                hh_apply_q(Qd, qtau, D, RP8, Qm, RP8);
                free(Qd); free(qtau); free(qrd);
                float *Q32 = (float *)xmalloc((size_t)D * RP8 * 4);
                for (size_t i2 = 0; i2 < (size_t)D * RP8; i2++) Q32[i2] = (float)Qm[i2];
                float *B32 = (float *)xmalloc((size_t)RP8 * F * 4);
                mm32(1, 0, RP8, F, D, Q32, RP8, Ms, F, B32, F);   /* B = Qᵀ·Ms */
                free(Q32); free(Ms);
                double *Bd = (double *)xmalloc((size_t)RP8 * F * sizeof(double));
                for (size_t i2 = 0; i2 < (size_t)RP8 * F; i2++) Bd[i2] = B32[i2];
                free(B32);
                double *Ub = NULL, *Sb = NULL, *Vb = NULL;
                zl_svd_lowrank(Bd, RP8, F, RP8, &Ub, &Sb, &Vb);   /* Ub[RP8,RP8] Sb[RP8] Vb[RP8,F] */
                free(Bd);
                double *Uw = (double *)xmalloc((size_t)D * ERF_R * sizeof(double));
                mm64(0, 0, D, ERF_R, RP8, Qm, RP8, Ub, RP8, Uw, ERF_R);   /* U = (Q·Ub)[:, :r] */
                free(Qm); free(Ub);
                double *Sw = (double *)xmalloc((size_t)ERF_R * sizeof(double));
                for (int c = 0; c < ERF_R; c++) Sw[c] = Sb[c];
                free(Sb);
                double *Vw = (double *)xmalloc((size_t)ERF_R * F * sizeof(double));
                for (int c = 0; c < ERF_R; c++) for (int j = 0; j < F; j++)
                    Vw[(size_t)c * F + j] = Vb[(size_t)c * F + j] / sh[j];  /* py: _V/_sh[None,:] */
                free(Vb); free(sh);

                /* _Gt/_Ge = (h@Vᵀ)·w ; 小 r×r 岭回归解 α */
                double *Gt = (double *)xmalloc((size_t)nft * ERF_R * sizeof(double));
                double *Ge = (double *)xmalloc((size_t)nfe * ERF_R * sizeof(double));
                mm64(0, 1, nft, ERF_R, F, ht, F, Vw, F, Gt, ERF_R);
                mm64(0, 1, nfe, ERF_R, F, he, F, Vw, F, Ge, ERF_R);
                free(ht); free(he);
                for (int j = 0; j < nft; j++) { double w = pw[ftr[j]];
                    for (int c = 0; c < ERF_R; c++) Gt[(size_t)j * ERF_R + c] *= w; }
                for (int j = 0; j < nfe; j++) { double w = pw[fev[j]];
                    for (int c = 0; c < ERF_R; c++) Ge[(size_t)j * ERF_R + c] *= w; }

                double *GtG = (double *)xmalloc((size_t)ERF_R * ERF_R * sizeof(double));
                mm64(1, 0, ERF_R, ERF_R, nft, Gt, ERF_R, Gt, ERF_R, GtG, ERF_R);
                double *UtU = (double *)xmalloc((size_t)ERF_R * ERF_R * sizeof(double));
                mm64(1, 0, ERF_R, ERF_R, D, Uw, ERF_R, Uw, ERF_R, UtU, ERF_R);
                double *Am2 = (double *)xmalloc((size_t)ERF_R * ERF_R * sizeof(double));
                for (int a = 0; a < ERF_R; a++) for (int b = 0; b < ERF_R; b++)
                    Am2[(size_t)a * ERF_R + b] = GtG[(size_t)a * ERF_R + b] * UtU[(size_t)a * ERF_R + b] * Sw[a] * Sw[b];
                free(GtG); free(UtU);
                /* _b = S ⊙ einsum("ni,nd,di->i", Gt, Rr, U) = S_i · Σ_n Gt[n,i]·(Rr[n,:]·U[:,i]) */
                double *Pru = (double *)xmalloc((size_t)nft * ERF_R * sizeof(double));
                { double *Rrow = (double *)xmalloc((size_t)nft * D * sizeof(double));
                  for (int j = 0; j < nft; j++) memcpy(Rrow + (size_t)j * D, ER + (size_t)prow[ftr[j]] * D, (size_t)D * sizeof(double));
                  mm64(0, 0, nft, ERF_R, D, Rrow, D, Uw, ERF_R, Pru, ERF_R);
                  free(Rrow); }
                double *bv = (double *)xmalloc((size_t)ERF_R * sizeof(double));
                for (int c = 0; c < ERF_R; c++) {
                    double s = 0;
                    for (int j = 0; j < nft; j++) s += Gt[(size_t)j * ERF_R + c] * Pru[(size_t)j * ERF_R + c];
                    bv[c] = Sw[c] * s;
                }
                free(Pru);
                double atr = 0;
                for (int c = 0; c < ERF_R; c++) atr += Am2[(size_t)c * ERF_R + c];
                for (int c = 0; c < ERF_R; c++) Am2[(size_t)c * ERF_R + c] += atr / ERF_R + 1e-12;
                if (lu_solve(Am2, ERF_R, bv)) { free(Am2); free(bv); free(Gt); free(Ge); free(Uw); free(Sw); free(Vw); continue; }
                free(Am2);

                /* _ct/_ce = (G ⊙ (α·S)) @ Uᵀ */
                double *aS = (double *)xmalloc((size_t)ERF_R * sizeof(double));
                for (int c = 0; c < ERF_R; c++) aS[c] = bv[c] * Sw[c];
                double *GtS = (double *)xmalloc((size_t)nft * ERF_R * sizeof(double));
                for (int j = 0; j < nft; j++) for (int c = 0; c < ERF_R; c++)
                    GtS[(size_t)j * ERF_R + c] = Gt[(size_t)j * ERF_R + c] * aS[c];
                double *GeS = (double *)xmalloc((size_t)nfe * ERF_R * sizeof(double));
                for (int j = 0; j < nfe; j++) for (int c = 0; c < ERF_R; c++)
                    GeS[(size_t)j * ERF_R + c] = Ge[(size_t)j * ERF_R + c] * aS[c];
                free(aS); free(Gt); free(Ge);
                double *ct = (double *)xmalloc((size_t)nft * D * sizeof(double));
                double *ce = (double *)xmalloc((size_t)nfe * D * sizeof(double));
                mm64(0, 1, nft, D, ERF_R, GtS, ERF_R, Uw, ERF_R, ct, D);
                mm64(0, 1, nfe, D, ERF_R, GeS, ERF_R, Uw, ERF_R, ce, D);
                free(GtS); free(GeS);

                /* token 能量门: 按 _ct 能量降序累积收益, 取累积最大处的能量当 τ */
                dsort_t *en = (dsort_t *)xmalloc((size_t)nft * sizeof(dsort_t));
                double *ben = (double *)xmalloc((size_t)nft * sizeof(double));
                for (int j = 0; j < nft; j++) {
                    const double *c2 = ct + (size_t)j * D, *r2 = ER + (size_t)prow[ftr[j]] * D;
                    double e2 = 0, dp = 0;
                    for (int d3 = 0; d3 < D; d3++) { e2 += c2[d3] * c2[d3]; dp += r2[d3] * c2[d3]; }
                    en[j].v = e2; en[j].i = j;
                    ben[j] = 2.0 * dp - e2;
                }
                dsort_t *oi = (dsort_t *)xmalloc((size_t)nft * sizeof(dsort_t));
                memcpy(oi, en, (size_t)nft * sizeof(dsort_t));
                qsort(oi, (size_t)nft, sizeof(dsort_t), cmp_dsort_desc);
                double cum = 0, best = -INFINITY; int m = 0;
                for (int j = 0; j < nft; j++) { cum += ben[oi[j].i]; if (cum > best) { best = cum; m = j + 1; } }
                free(ben);
                if (best <= 0) { free(en); free(oi); free(ct); free(ce); free(Uw); free(Sw); free(Vw); free(bv); continue; }
                double tau = oi[m - 1].v;
                free(oi);
                for (int j = 0; j < nft; j++) if (en[j].v < tau) memset(ct + (size_t)j * D, 0, (size_t)D * sizeof(double));
                free(en);
                for (int j = 0; j < nfe; j++) {
                    double e2 = 0; const double *c2 = ce + (size_t)j * D;
                    for (int d3 = 0; d3 < D; d3++) e2 += c2[d3] * c2[d3];
                    if (e2 < tau) memset(ce + (size_t)j * D, 0, (size_t)D * sizeof(double));
                }
                double chk = 0;
                for (int j = 0; j < nft; j++) {
                    const double *c2 = ct + (size_t)j * D, *r2 = ER + (size_t)prow[ftr[j]] * D;
                    for (int d3 = 0; d3 < D; d3++) chk += 2.0 * r2[d3] * c2[d3] - c2[d3] * c2[d3];
                }
                if (chk <= 0) { free(ct); free(ce); free(Uw); free(Sw); free(Vw); free(bv); continue; }

                for (int j = 0; j < nft; j++) {
                    double *r2 = ER + (size_t)prow[ftr[j]] * D;
                    const double *c2 = ct + (size_t)j * D;
                    for (int d3 = 0; d3 < D; d3++) r2[d3] -= c2[d3];
                }
                for (int j = 0; j < nfe; j++) {
                    int po = evpos[prow[fev[j]]];
                    if (po < 0) continue;
                    double *p2 = pred + (size_t)po * D;
                    const double *c2 = ce + (size_t)j * D;
                    for (int d3 = 0; d3 < D; d3++) p2[d3] += c2[d3];
                }
                free(ct); free(ce);

                /* 载荷: <If>(e,τ) + U16=(U⊙S)[D,r] + V16=(α·V)[r,F], 都是 f64 直接舍 f16 */
                size_t one = 8 + (size_t)D * ERF_R * 2 + (size_t)ERF_R * F * 2;
                epay = (uint8_t *)realloc(epay, epay_len + one);
                if (!epay) die("realloc ERF 载荷");
                uint8_t *dst = epay + epay_len; epay_len += one;
                uint32_t e32 = (uint32_t)e; float tau32 = (float)tau;
                memcpy(dst, &e32, 4); memcpy(dst + 4, &tau32, 4);
                uint16_t *U16e = (uint16_t *)(dst + 8), *V16e = U16e + (size_t)D * ERF_R;
                for (int r2 = 0; r2 < D; r2++) for (int c = 0; c < ERF_R; c++)
                    U16e[(size_t)r2 * ERF_R + c] = f64_to_f16(Uw[(size_t)r2 * ERF_R + c] * Sw[c]);
                for (int c = 0; c < ERF_R; c++) for (int j = 0; j < F; j++)
                    V16e[(size_t)c * F + j] = f64_to_f16(bv[c] * Vw[(size_t)c * F + j]);
                free(Uw); free(Sw); free(Vw); free(bv);
                ne++;
            }
            free(ftr); free(fev); free(eidx); free(ecsr);

            if (ne) {
                double e0a = 0, e1a = 0;
                for (size_t i2 = 0; i2 < (size_t)nev * D; i2++) {
                    e0a += Rev0[i2] * Rev0[i2];
                    double d3 = Rev0[i2] - pred[i2]; e1a += d3 * d3;
                }
                ERF_GAIN = 1.0 - e1a / (e0a > 1e-18 ? e0a : 1e-18);
                if (ERF_GAIN > 0.002) {
                    uint8_t *hdr8 = (uint8_t *)xmalloc(8 + epay_len);
                    uint32_t u32ne = (uint32_t)ne; uint16_t u16r = (uint16_t)ERF_R, u16z = 0;
                    memcpy(hdr8, &u32ne, 4); memcpy(hdr8 + 4, &u16r, 2); memcpy(hdr8 + 6, &u16z, 2);
                    memcpy(hdr8 + 8, epay, epay_len);
                    ERF_ADD = make_rec("zl.ERF", hdr8, 8 + epay_len, &ERF_LEN);
                    free(hdr8);
                    ERF_NE = ne;
                }
            }
            double tot = eff + (ERF_GAIN > 0 ? ERF_GAIN : 0.0) * (1 - eff);
            printf("  L%d ERF死层部件: 专家=%d 残差挽回=%.1f%% 组合总计=%.1f%% → %s\n",
                   L, ne, ERF_GAIN * 100, tot * 100, ERF_ADD ? "注入" : "不过闸");
            free(epay); free(pred); free(Rev0); free(ER); free(trm); free(evm); free(evpos);
            munmap((void *)eblob, ebsz);
            free(esc);
        }
    }

