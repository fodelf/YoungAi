int main(int argc, char **argv) {
    const char *cap = NULL, *outd = NULL, *layers = NULL;
    int threads = 20, KMAX = 1024, nfit_arg = 0, feat_yq = 0;
    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "--cap") || !strcmp(argv[i], "--cap-dir")) && i + 1 < argc) cap = argv[++i];   /* --cap-dir=引擎同名别名 */
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outd = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kmax") && i + 1 < argc) KMAX = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nfit") && i + 1 < argc) nfit_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--feat") && i + 1 < argc) feat_yq = !strcmp(argv[++i], "yq");
    }
    if (!cap || !outd || !layers) {
        fprintf(stderr, "用法: amp_solve --cap capnpy目录 --out zrec目录 --layers a-b [--threads N] [--kmax K]\n");
        return 2;
    }
    int l0 = 0, l1 = 0;
    if (sscanf(layers, "%d-%d", &l0, &l1) != 2) { l0 = l1 = atoi(layers); }

    const double LAMS[] = {1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0};
    const int NLAM = 7;
    const int KS[] = {16, 32, 64, 128, 256, 384, 512, 768, 1024};
    const int NKS = 9;

    for (int L = l0; L <= l1; L++) {
        time_t t0 = time(NULL);
        char p[1024]; npy_meta mx, mq, mf;
        /* --feat yq: 特征源换 ŷ(部署字节 routed_out) —— 四支柱"专家输出预测"。
         * 引擎侧落地 = 把 routed_out 绑为 zchain 特征输入(solve_rrr.h feat_yhat 同型)。 */
        snprintf(p, sizeof p, feat_yq ? "%s/obase_v3_L%d.npy" : "%s/ffn_in_L%d.npy", cap, L);
        float *X0 = npy_read_f32(p, &mx);
        snprintf(p, sizeof p, "%s/obase_v3_L%d.npy", cap, L);
        float *yq = npy_read_f32(p, &mq);
        snprintf(p, sizeof p, "%s/routed_L%d.npy", cap, L);
        float *yfp = npy_read_f32(p, &mf);
        if (!X0 || !yq || !yfp || mx.ndim != 2 || mx.shape[1] != DM ||
            mq.shape[0] != mx.shape[0] || mf.shape[0] != mx.shape[0]) {
            fprintf(stderr, "L%d: cap 张量缺/形状错\n", L);
            free(X0); free(yq); free(yfp); continue;
        }
        const int S = (int)mx.shape[0], D = DM, DIN = 3 * DM;
        const int NFIT = nfit_arg > 0 ? nfit_arg : S * 8 / 10;
        const int NEV = S - NFIT, NA = 2 * NFIT;

        /* φ 提升整表 [S, DIN] */
        float *X = malloc((size_t)S * DIN * sizeof(float));
        if (!X) { fprintf(stderr, "L%d OOM X\n", L); return 3; }
        { phi_ctx pc = {X0, X, D}; parallel_for(S, threads, phi_worker, &pc); }
        free(X0);

        /* eps / R / colw (训练段统计) */
        double *msq = calloc(D, sizeof(double));
        for (int t = 0; t < NFIT; t++) {
            const float *y = yq + (size_t)t * D;
            for (int j = 0; j < D; j++) msq[j] += (double)y[j] * y[j];
        }
        float *eps = malloc(D * sizeof(float));
        for (int j = 0; j < D; j++) eps[j] = (float)(sqrt(msq[j] / NFIT) * 1e-2 + 1e-12);
        float *R = malloc((size_t)S * D * sizeof(float));
        for (int t = 0; t < S; t++) {
            const float *y = yq + (size_t)t * D, *yf = yfp + (size_t)t * D;
            float *r = R + (size_t)t * D;
            for (int j = 0; j < D; j++) {
                float dh = yf[j] - y[j];
                r[j] = dh * y[j] / (y[j] * y[j] + eps[j] * eps[j]);
            }
        }
        /* L_classify 感知列权 = y* 的 per-dim 判别方差权(训练段), 归一到均值 1。
         * 旧版 std(R)·rms(yq) 是残差空间权 = 修残差框架, 已废(2026-08-23)。 */
        double *rm = calloc(D, sizeof(double)), *rv = calloc(D, sizeof(double));
        for (int t = 0; t < NFIT; t++) {
            const float *yf = yfp + (size_t)t * D;
            for (int j = 0; j < D; j++) { rm[j] += yf[j]; rv[j] += (double)yf[j] * yf[j]; }
        }
        double *colw = malloc(D * sizeof(double));
        double cwm = 0;
        for (int j = 0; j < D; j++) {
            double mean = rm[j] / NFIT, var = rv[j] / NFIT - mean * mean;
            colw[j] = sqrt((var > 0 ? var : 0) + 1e-12);
            cwm += colw[j];
        }
        cwm /= D;
        for (int j = 0; j < D; j++) colw[j] /= cwm;
        /* L_align 行权: α_t = 1/(‖y*_t‖ + ε‖·‖均值) —— √权乘进增广行两侧, MSE→cos 几何 */
        double *ralpha = malloc((size_t)S * sizeof(double));
        { double nm = 0;
          for (int t = 0; t < S; t++) {
              const float *yf = yfp + (size_t)t * D;
              double ss = 0;
              for (int j = 0; j < D; j++) ss += (double)yf[j] * yf[j];
              ralpha[t] = sqrt(ss); nm += ralpha[t];
          }
          nm /= S;
          for (int t = 0; t < S; t++) ralpha[t] = 1.0 / (ralpha[t] + 1e-3 * nm); }
        free(msq); free(rm); free(rv); free(eps);

        /* dither 增广: Xa = [X_tr; X_tr + 0.04·rms_row·N] ; Ra = [R_tr·colw] ×2 */
        float *Xa = malloc((size_t)NA * DIN * sizeof(float));
        float *Ra = malloc((size_t)NA * D * sizeof(float));
        if (!Xa || !Ra) { fprintf(stderr, "L%d OOM Xa\n", L); return 3; }
        memcpy(Xa, X, (size_t)NFIT * DIN * sizeof(float));
        {   /* 行独立种子 → 可并行(原单线程 8000 万次 Box-Muller ≈ 4s) */
            dith_ctx dc = {X, Xa + (size_t)NFIT * DIN, DIN, (uint64_t)L};
            parallel_for(NFIT, threads, dith_worker, &dc);
        }
        for (int t = 0; t < NFIT; t++) {
            const float *r = R + (size_t)t * D;
            float *ra = Ra + (size_t)t * D, *rb = Ra + (size_t)(NFIT + t) * D;
            const double w = ralpha[t];
            for (int j = 0; j < D; j++) ra[j] = rb[j] = (float)(r[j] * colw[j] * w);
        }

        /* scale = rms(Xa) */
        double ssq = 0;
        for (size_t i = 0; i < (size_t)NA * DIN; i += 97) ssq += (double)Xa[i] * Xa[i];  /* 步进抽样, 均值稳 */
        float scale = (float)sqrt(ssq / (((size_t)NA * DIN + 96) / 97));

        /* 随机基 Vt/At [KMAX, din] 行主(转置存, dot 连续) */
        float *Vt = malloc((size_t)KMAX * DIN * sizeof(float));
        float *At = malloc((size_t)KMAX * DIN * sizeof(float));
        {   /* 每基行独立种子 → 并行(原单线程 5000 万次 Box-Muller) */
            base_ctx bv = {Vt, DIN, 7};  parallel_for(KMAX, threads, base_worker, &bv);
            base_ctx ba = {At, DIN, 11}; parallel_for(KMAX, threads, base_worker, &ba);
        }

        /* 特征: Za [NA, K], Ze [NEV, K] */
        float *Za = malloc((size_t)NA * KMAX * sizeof(float));
        float *Ze = malloc((size_t)NEV * KMAX * sizeof(float));
        if (!Za || !Ze) { fprintf(stderr, "L%d OOM Z\n", L); return 3; }
        time_t tf0 = time(NULL);
        feat_ctx fc = {Xa, Vt, At, Za, DIN, KMAX, 1.0f / scale};
        parallel_for(NA, threads, feat_worker, &fc);
        feat_ctx fe = {X + (size_t)NFIT * DIN, Vt, At, Ze, DIN, KMAX, 1.0f / scale};
        parallel_for(NEV, threads, feat_worker, &fe);
        free(Xa);
        /* 行归一(L_align): 加权 LS 的 α 权 —— Za 行与目标行(Ra, 已乘)各带一次 α ⇒
         * 归一方程 (Zᵀdiag(α²)Z)U = Zᵀdiag(α²)R̃ = cos 几何的行归一化。 */
        for (int t = 0; t < NA; t++) {
            const double w = ralpha[t < NFIT ? t : t - NFIT];
            float *zr = Za + (size_t)t * KMAX;
            for (int k = 0; k < KMAX; k++) zr[k] = (float)(zr[k] * w);
        }
        time_t tf1 = time(NULL);

        /* ZtZ / ZtR */
        double *ZtZ = calloc((size_t)KMAX * KMAX, sizeof(double));
        double *ZtR = calloc((size_t)KMAX * D, sizeof(double));
        {
            int nth = threads > 20 ? 20 : threads;
            double *pz[20], *pr2[20];
            for (int i = 0; i < nth; i++) {
                pz[i] = calloc((size_t)KMAX * KMAX, sizeof(double));
                pr2[i] = calloc((size_t)KMAX * D, sizeof(double));
            }
            gram_ctx gc = {Za, Ra, ZtZ, ZtR, pz, pr2, NA, KMAX, D, nth};
            parallel_for(NA, nth, gram_worker, &gc);
            for (int i = 0; i < nth; i++) {
                for (size_t q = 0; q < (size_t)KMAX * KMAX; q++) ZtZ[q] += pz[i][q];
                for (size_t q = 0; q < (size_t)KMAX * D; q++) ZtR[q] += pr2[i][q];
                free(pz[i]); free(pr2[i]);
            }
        }
        for (int a = 0; a < KMAX; a++)                    /* 对称补全 */
            for (int b = 0; b < a; b++) ZtZ[(size_t)a * KMAX + b] = ZtZ[(size_t)b * KMAX + a];
        free(Za); free(Ra);
        double trZ = 0;
        for (int a = 0; a < KMAX; a++) trZ += ZtZ[(size_t)a * KMAX + a];

        /* 基线对齐度: mean cos(yq, y*) on held(放大器要抬的就是这个数) */
        double cos0 = 0;
        for (int t = NFIT; t < S; t++) {
            const float *y = yq + (size_t)t * D, *yf = yfp + (size_t)t * D;
            double d = 0, na = 0, nb = 0;
            for (int j = 0; j < D; j++) { d += (double)y[j] * yf[j]; na += (double)y[j] * y[j]; nb += (double)yf[j] * yf[j]; }
            cos0 += (na > 0 && nb > 0) ? d / (sqrt(na) * sqrt(nb)) : 0;
        }
        cos0 /= NEV;

        /* λ × k 网格 */
        double best_rec = -2; int best_lam = -1, best_k = 0;   /* 判据=held mean cos */
        double *Ubest = malloc((size_t)KMAX * D * sizeof(double));
        double *A_ch = malloc((size_t)KMAX * KMAX * sizeof(double));
        double *U = malloc((size_t)KMAX * D * sizeof(double));
        double *Uw = malloc((size_t)KMAX * D * sizeof(double));
        float *G = malloc((size_t)NEV * D * sizeof(float));
        double part[64];
        for (int li = 0; li < NLAM; li++) {
            memcpy(A_ch, ZtZ, (size_t)KMAX * KMAX * sizeof(double));
            double ridge = LAMS[li] * trZ / KMAX + 1e-10;
            for (int a = 0; a < KMAX; a++) A_ch[(size_t)a * KMAX + a] += ridge;
            memcpy(U, ZtR, (size_t)KMAX * D * sizeof(double));
            if (chol_solve(A_ch, KMAX, U, D, threads)) { fprintf(stderr, "L%d λ=%g chol 失败\n", L, LAMS[li]); continue; }
            for (int a = 0; a < KMAX; a++)
                for (int j = 0; j < D; j++) Uw[(size_t)a * D + j] = U[(size_t)a * D + j] / colw[j];
            memset(G, 0, (size_t)NEV * D * sizeof(float));
            int kprev = 0;
            for (int ki = 0; ki < NKS && KS[ki] <= KMAX; ki++) {
                acc_ctx ac = {Ze, Uw, G, KMAX, D, kprev, KS[ki]};
                parallel_for(NEV, threads, acc_worker, &ac);
                kprev = KS[ki];
                rec_ctx rc = {yq + (size_t)NFIT * D, yfp + (size_t)NFIT * D, G, part, D, threads, NEV};
                memset(part, 0, sizeof(double) * 64);
                parallel_for(NEV, threads, rec_worker, &rc);
                double csum = 0;
                for (int i = 0; i < 64; i++) csum += part[i];
                double rec = csum / NEV;               /* held mean cos(ŷ, y*) */
                if (rec > best_rec) {
                    best_rec = rec; best_lam = li; best_k = KS[ki];
                    memcpy(Ubest, U, (size_t)KMAX * D * sizeof(double));
                }
            }
        }
        free(A_ch); free(U); free(Uw); free(G); free(ZtZ); free(ZtR); free(Ze);
        fprintf(stderr, "L%d 段耗时: 准备 %lds 特征 %lds gram+网格 %lds\n", L,
                (long)(tf0 - t0), (long)(tf1 - tf0), (long)(time(NULL) - tf1));

        /* zrec 写出 (与 amp_solve.py hdr()/payload 逐字节同构) */
        snprintf(p, sizeof p, "%s/zrec_L%02d.bin", outd, L);
        FILE *f = fopen(p, "wb");
        if (!f) { fprintf(stderr, "L%d: %s 打不开\n", L, p); return 4; }
        uint8_t hdr[DS4_AMP_REC_HDR]; memset(hdr, 0, sizeof hdr);
        if (best_rec <= cos0 + 2e-4 || best_lam < 0) {   /* 层闸: held 对齐度无增益 */
            memcpy(hdr, "zl.AMP", 6);
            uint64_t psz = 0; memcpy(hdr + DS4_AMP_REC_OFF_PSZ, &psz, 8);
            int32_t one = 1; memcpy(hdr + DS4_AMP_REC_OFF_VD, &one, 4);
            fwrite(hdr, 1, DS4_AMP_REC_HDR, f); fclose(f);
            printf("★L%d 放大器(C): held对齐 %.4f→%.4f 无增益 → 层闸 | %lds\n",
                   L, cos0, best_rec, (long)(time(NULL) - t0));
        } else {
            const int k = best_k;
            memcpy(hdr, "zl.AMPD", 7);
            uint64_t psz = DS4_AMP_OP_HDR + 2 * DS4_AMP_AMPD_ELEMS((size_t)k, DIN, D);   /* zl.AMPD: A|U|V */
            memcpy(hdr + DS4_AMP_REC_OFF_PSZ, &psz, 8);
            int32_t one = 1; memcpy(hdr + DS4_AMP_REC_OFF_VD, &one, 4);
            fwrite(hdr, 1, DS4_AMP_REC_HDR, f);
            uint32_t ku = (uint32_t)k, dinu = (uint32_t)DIN, du = (uint32_t)D;
            fwrite(&ku, 4, 1, f); fwrite(&scale, 4, 1, f); fwrite(&dinu, 4, 1, f); fwrite(&du, 4, 1, f);
            uint16_t *h16 = malloc((size_t)DIN * k * sizeof(uint16_t));
            /* A: [DIN, k] 列 c = At 行 c */
            for (int j = 0; j < DIN; j++)
                for (int c = 0; c < k; c++) h16[(size_t)j * k + c] = f32_to_f16(At[(size_t)c * DIN + j]);
            fwrite(h16, 2, (size_t)DIN * k, f);
            /* U: [D, k] = (Ubest[:k]/colw)ᵀ */
            for (int j = 0; j < D; j++)
                for (int c = 0; c < k; c++)
                    h16[(size_t)j * k + c] = f32_to_f16((float)(Ubest[(size_t)c * D + j] / colw[j]));
            fwrite(h16, 2, (size_t)D * k, f);
            /* V: [DIN, k] 列 c = Vt 行 c */
            for (int j = 0; j < DIN; j++)
                for (int c = 0; c < k; c++) h16[(size_t)j * k + c] = f32_to_f16(Vt[(size_t)c * DIN + j]);
            fwrite(h16, 2, (size_t)DIN * k, f);
            free(h16); fclose(f);
            printf("★L%d 放大器(C): held对齐 cos %.4f→%.4f (Δ%+.4f) @λ=%.0f k_L=%d 体积 %.1fMB | %lds\n",
                   L, cos0, best_rec, best_rec - cos0, LAMS[best_lam], k,
                   (double)psz / (1 << 20), (long)(time(NULL) - t0));
        }
        fflush(stdout);
        free(X); free(yq); free(yfp); free(R); free(colw); free(ralpha); free(Vt); free(At); free(Ubest);
    }
    return 0;
}
