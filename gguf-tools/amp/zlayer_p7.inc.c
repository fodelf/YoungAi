    /* ---------------- 注入载荷 ---------------- */
    uint8_t *add = NULL; size_t add_len = 0; int nrec_add = 0;
    if (ADDON) {
        /* ★叠加式合并注入★: 既有记录(GE, z_old)与本轮新解合并成【单条】, 引擎按"末条胜出"
         * 读到的就是合并后的整体 —— 所以 GE 直接相乘, 两段低秩拼成一个矩阵再重新截断:
         *   M = [V_old·z_old | V_new·S_new] · [U_old | U_new]ᵀ = Pc·Qcᵀ
         * 对 Pc/Qc 各做一次 QR, 只对 n×n 的小核 Rp·Rqᵀ 求 SVD, 再把基乘回去 —— 等价于
         * 对 M 直接做截断 SVD, 但不用把 DIW×D 的稠密 M 摆出来。 */
        float ge_m[NEXP];
        for (int e = 0; e < NEXP; e++) {
            float gn = USE_GE ? gf[e] : 1.0f;
            float gb2 = ge_old ? ge_old[e] : 1.0f;
            ge_m[e] = gb2 * gn;
        }
        uint16_t ge_m16[NEXP];
        for (int e = 0; e < NEXP; e++) ge_m16[e] = f64_to_f16((double)ge_m[e]);
        const int ftaW = (K > 0 && !strcmp(FORM, "ftA"));
        const int DIW = ((zo_z && zo_di == 3 * D) || ftaW) ? 3 * D : D;
        const int kold = zo_z ? zo_k0 : 0, knew = (K > 0) ? K : 0;
        const int ncol = kold + knew;
        if (ncol == 0) {
            add = make_rec("bf.GE", ge_m16, sizeof ge_m16, &add_len);
            nrec_add = 1;
        } else {
            double *Pc = (double *)xcalloc((size_t)DIW * ncol, sizeof(double));
            double *Qc = (double *)xcalloc((size_t)D * ncol, sizeof(double));
            if (kold) {                                   /* P=lift(V_old·z_old), Q=U_old */
                for (int i = 0; i < zo_di; i++) for (int c = 0; c < kold; c++)
                    Pc[(size_t)i * ncol + c] = (double)zo_V[(size_t)i * zo_k0 + c] * (double)zo_z[c];
                for (int j = 0; j < D; j++) for (int c = 0; c < kold; c++)
                    Qc[(size_t)j * ncol + c] = zo_U[(size_t)j * zo_k0 + c];
            }
            if (knew) {                                   /* P=lift(A[:,:K]·S), Q=Bt[:K].T */
                int dinn = ftaW ? DIN3 : D, rr = ftaW ? rfta : rlin;
                const double *AA = ftaW ? Af : A, *SS = ftaW ? Sf : S, *BB = ftaW ? Bf : Bt;
                for (int i = 0; i < dinn; i++) for (int c = 0; c < knew; c++)
                    Pc[(size_t)i * ncol + kold + c] = AA[(size_t)i * rr + c] * SS[c];
                for (int j = 0; j < D; j++) for (int c = 0; c < knew; c++)
                    Qc[(size_t)j * ncol + kold + c] = BB[(size_t)c * D + j];
            }
            double *tauP = (double *)xmalloc((size_t)ncol * sizeof(double));
            double *rdP = (double *)xmalloc((size_t)ncol * sizeof(double));
            double *tauQ = (double *)xmalloc((size_t)ncol * sizeof(double));
            double *rdQ = (double *)xmalloc((size_t)ncol * sizeof(double));
            hh_qr(Pc, DIW, ncol, tauP, rdP);
            hh_qr(Qc, D, ncol, tauQ, rdQ);
            double *Rp = (double *)xcalloc((size_t)ncol * ncol, sizeof(double));
            double *Rq = (double *)xcalloc((size_t)ncol * ncol, sizeof(double));
            for (int i = 0; i < ncol; i++) {
                Rp[(size_t)i * ncol + i] = rdP[i]; Rq[(size_t)i * ncol + i] = rdQ[i];
                for (int j = i + 1; j < ncol; j++) {
                    Rp[(size_t)i * ncol + j] = Pc[(size_t)i * ncol + j];
                    Rq[(size_t)i * ncol + j] = Qc[(size_t)i * ncol + j];
                }
            }
            free(rdP); free(rdQ);
            double *Mm = (double *)xmalloc((size_t)ncol * ncol * sizeof(double));
            mm64(0, 1, ncol, ncol, ncol, Rp, ncol, Rq, ncol, Mm, ncol);   /* Rp·Rqᵀ */
            free(Rp); free(Rq);
            /* py 用全 SVD 再数 Sm>1e-8 并封顶 1024。奇异值降序 ⇒ "全谱里 >1e-8 的个数封顶
             * 1024" 恒等于 "前 min(n,1024) 个里 >1e-8 的个数", 所以只求前 rm 个就够。 */
            int rm = ncol < 1088 ? ncol : 1088;
            double *Am3 = NULL, *Sm3 = NULL, *Bm3 = NULL;
            zl_svd_lowrank(Mm, ncol, ncol, rm, &Am3, &Sm3, &Bm3);
            free(Mm);
            int cap = ncol < 1024 ? ncol : 1024;
            int km = 0;
            for (int c = 0; c < cap && c < rm; c++) if (Sm3[c] > 1e-8) km++;
            if (km == 0) km = 1;
            double *Vm = (double *)xcalloc((size_t)DIW * km, sizeof(double));
            for (int i = 0; i < ncol; i++) for (int c = 0; c < km; c++)
                Vm[(size_t)i * km + c] = Am3[(size_t)i * rm + c];
            hh_apply_q(Pc, tauP, DIW, ncol, Vm, km);                      /* Vm = Qp·Am[:, :km] */
            double *Um = (double *)xcalloc((size_t)D * km, sizeof(double));
            for (int i = 0; i < ncol; i++) for (int c = 0; c < km; c++)
                Um[(size_t)i * km + c] = Bm3[(size_t)c * ncol + i];       /* Bm.T[:, :km] */
            hh_apply_q(Qc, tauQ, D, ncol, Um, km);                        /* Um = Qq·Bm.T[:, :km] */
            free(Pc); free(Qc); free(tauP); free(tauQ); free(Am3); free(Bm3);

            size_t rl1; uint8_t *r1 = make_rec("bf.GE", ge_m16, sizeof ge_m16, &rl1);
            size_t nh = (size_t)km + (size_t)D * km + (size_t)DIW * km;
            size_t psz = 16 + nh * 2;
            uint8_t *pay = (uint8_t *)xmalloc(psz);
            uint32_t u32k = (uint32_t)km, u32di = (uint32_t)DIW, u32do = (uint32_t)D;
            float tr05 = 0.5f;
            memcpy(pay, &u32k, 4); memcpy(pay + 4, &tr05, 4);
            memcpy(pay + 8, &u32di, 4); memcpy(pay + 12, &u32do, 4);
            uint16_t *h = (uint16_t *)(pay + 16);
            for (int c = 0; c < km; c++) h[c] = f64_to_f16(Sm3[c]);
            uint16_t *U16 = h + km, *V16 = h + km + (size_t)D * km;
            for (size_t i = 0; i < (size_t)D * km; i++) U16[i] = f64_to_f16(Um[i]);
            for (size_t i = 0; i < (size_t)DIW * km; i++) V16[i] = f64_to_f16(Vm[i]);
            free(Sm3); free(Um); free(Vm);
            size_t rl2; uint8_t *r2 = make_rec("zl.RRR", pay, psz, &rl2);
            free(pay);
            add_len = rl1 + rl2;
            add = (uint8_t *)xmalloc(add_len);
            memcpy(add, r1, rl1); memcpy(add + rl1, r2, rl2);
            free(r1); free(r2);
            nrec_add = 2;
        }
    } else if (USE_GE) {
        size_t rl; uint8_t *r = make_rec("bf.GE", ge16, sizeof ge16, &rl);
        add = (uint8_t *)realloc(add, add_len + rl); if (!add) die("realloc");
        memcpy(add + add_len, r, rl); add_len += rl; free(r);
        nrec_add++;
    }
    if (!ADDON && K > 0) {
        int ftaW = !strcmp(FORM, "ftA");
        int din = ftaW ? DIN3 : D, r = ftaW ? rfta : rlin;
        const double *AA = ftaW ? Af : A, *SS = ftaW ? Sf : S, *BB = ftaW ? Bf : Bt;
        size_t nh = (size_t)K + (size_t)D * K + (size_t)din * K;
        size_t psz = 16 + nh * 2;
        uint8_t *pay = (uint8_t *)xmalloc(psz);
        uint32_t u32k = (uint32_t)K, u32di = (uint32_t)din, u32do = (uint32_t)D;
        /* ★tr 槽写 0.5★ 引擎 type6 拿它当信任域上限(‖z 出力‖ ≤ tr·‖routed‖), 不是"关闭夹持"
         * 的大数 —— 今晨定的契约, 写 1e6 等于事实上无夹持。 */
        float tr05 = 0.5f;
        memcpy(pay, &u32k, 4); memcpy(pay + 4, &tr05, 4);
        memcpy(pay + 8, &u32di, 4); memcpy(pay + 12, &u32do, 4);
        uint16_t *h = (uint16_t *)(pay + 16);
        for (int c = 0; c < K; c++) h[c] = f64_to_f16(SS[c]);                       /* z[K] */
        uint16_t *U16 = h + K, *V16 = h + K + (size_t)D * K;
        for (int j = 0; j < D; j++) for (int c = 0; c < K; c++)
            U16[(size_t)j * K + c] = f64_to_f16(BB[(size_t)c * D + j]);             /* U=Bt[:K].T [D,K] */
        for (int i = 0; i < din; i++) for (int c = 0; c < K; c++)
            V16[(size_t)i * K + c] = f64_to_f16(AA[(size_t)i * r + c]);             /* V=A[:,:K] [din,K] */
        size_t rl; uint8_t *rec = make_rec("zl.RRR", pay, psz, &rl);
        free(pay);
        add = (uint8_t *)realloc(add, add_len + rl); if (!add) die("realloc");
        memcpy(add + add_len, rec, rl); add_len += rl; free(rec);
        nrec_add++;
    }

    char status[512]; snprintf(status, sizeof status, "解算完");
    if (INJ == 2) {
        snprintf(path, sizeof path, "%s/zrec_L%02d.bin", ld, L);
        FILE *zf = fopen(path, "wb");
        if (!zf) die("zrec 写不开: %s", path);
        if (eff <= GATE) {
            snprintf(status, sizeof status, "组合增益 %.1f%% ≤闸%.1f%% → 空 zrec(skip 标记)", comb * 100, GATE * 100);
        } else {
            if (add_len && fwrite(add, 1, add_len, zf) != add_len) die("zrec 写失败");
            snprintf(status, sizeof status, "zrec 落盘(+%d记录 %.1fMB)", nrec_add, add_len / 1048576.0);
        }
        fclose(zf);
    } else if (INJ) {
        char man[1300]; snprintf(man, sizeof man, "%s/zinject_manifest.txt", ld);
        char mlk[1400]; snprintf(mlk, sizeof mlk, "%s.lock", man);
        int lkfd = open(mlk, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lkfd < 0) die("账本锁打不开: %s", mlk);
        if (flock(lkfd, LOCK_EX)) die("账本上锁失败");     /* ★双路并发: 账本读写全程持锁 */
        int done = 0;
        FILE *mf = fopen(man, "r");
        if (mf) { char line[256];
            /* 空行/垃圾行不能当成 "0" —— 否则 L0 会被误判成已注入过而整层跳过 */
            while (fgets(line, sizeof line, mf)) {
                const char *q = line;
                while (*q == ' ' || *q == '\t') q++;
                if (!(*q == '-' || (*q >= '0' && *q <= '9'))) continue;
                if (atoi(q) == L) done = 1;
            }
            fclose(mf); }
        char dql[1300]; snprintf(dql, sizeof dql, "%s/dql_L%02d.bin", ld, L);
        if (ADDON) {
            /* ★叠加式落地★ 注意 py 把这一支放在 `elif L in done` 【前面】—— 叠加式不看
             * "已注入过", 它本来就是要覆盖上一轮贪心记录的。
             * 闸拒 = 零动作(贪心原记录原样留着); 落地 = 先按账本截回裸底座(去掉旧记录)
             * 再写合并记录, 账本不重复记。 */
            if (eff <= GATE) {
                snprintf(status, sizeof status, "Δ增益 %.1f%% ≤闸%.1f%% → 保留贪心原记录(叠加闸)", eff * 100, GATE * 100);
            } else {
                long long ent_sz = -1; uint32_t ent_n0 = 0; int have_ent = 0;
                mf = fopen(man, "r");
                if (mf) { char line[256];
                    while (fgets(line, sizeof line, mf)) {
                        const char *q = line;
                        while (*q == ' ' || *q == '\t') q++;
                        if (!(*q == '-' || (*q >= '0' && *q <= '9'))) continue;
                        long a1 = 0, a2 = 0, a3 = 0;
                        if (sscanf(q, "%ld %ld %ld", &a1, &a2, &a3) != 3) continue;
                        if (a1 == L && a2 > 0) { ent_sz = a2; ent_n0 = (uint32_t)a3; have_ent = 1; }
                    }
                    fclose(mf); }
                if (have_ent) {
                    FILE *df = fopen(dql, "r+b");
                    if (!df) die("dql 打不开(r+b): %s", dql);
                    if (ftruncate(fileno(df), (off_t)ent_sz)) die("dql 截回原账长度失败");
                    if (fseeko(df, 8, SEEK_SET) || fwrite(&ent_n0, 4, 1, df) != 1) die("dql nrec 回写失败");
                    if (fseeko(df, 0, SEEK_END)) die("dql seek end 失败");
                    if (add_len && fwrite(add, 1, add_len, df) != add_len) die("dql 追加失败");
                    uint32_t n1 = ent_n0 + (uint32_t)nrec_add;
                    if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
                    fclose(df);
                    snprintf(status, sizeof status, "合并注入完(+%d记录, 基于原账 %lld)", nrec_add, ent_sz);
                } else {
                    struct stat ds;
                    if (stat(dql, &ds)) die("dql 不存在: %s", dql);
                    long long osz = (long long)ds.st_size;
                    FILE *df = fopen(dql, "r+b");
                    if (!df) die("dql 打不开(r+b): %s", dql);
                    uint32_t n0;
                    if (fseeko(df, 8, SEEK_SET) || fread(&n0, 4, 1, df) != 1) die("dql nrec 读不到");
                    if (fseeko(df, 0, SEEK_END)) die("dql seek end 失败");
                    if (add_len && fwrite(add, 1, add_len, df) != add_len) die("dql 追加失败");
                    uint32_t n1 = n0 + (uint32_t)nrec_add;
                    if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
                    fclose(df);
                    mf = fopen(man, "a"); if (!mf) die("账本写不开");
                    fprintf(mf, "%d %lld %u\n", L, osz, n0); fclose(mf);
                    snprintf(status, sizeof status, "合并注入完(+%d记录, 新账 %lld)", nrec_add, osz);
                }
            }
        } else if (done) {
            snprintf(status, sizeof status, "已注入过, 跳过(回滚请按账本截断)");
        } else if (eff <= GATE) {
            if (ERF_ADD) {          /* ★死层部件接管★: z 被闸拒但 ERF 过闸 → 同账本注入 */
                struct stat ds;
                if (stat(dql, &ds)) die("dql 不存在: %s", dql);
                long long osz = (long long)ds.st_size;
                FILE *df = fopen(dql, "r+b");
                if (!df) die("dql 打不开(r+b): %s", dql);
                uint32_t n0;
                if (fseeko(df, 8, SEEK_SET) || fread(&n0, 4, 1, df) != 1) die("dql nrec 读不到");
                if (fseeko(df, 0, SEEK_END)) die("dql seek end 失败");
                if (fwrite(ERF_ADD, 1, ERF_LEN, df) != ERF_LEN) die("dql 追加失败");
                uint32_t n1 = n0 + 1;
                if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
                fclose(df);
                mf = fopen(man, "a"); if (!mf) die("账本写不开");
                fprintf(mf, "%d %lld %u\n", L, osz, n0); fclose(mf);
                snprintf(status, sizeof status, "ERF死层注入(+1记录 %.1fMB, held+%.1f%%, 专家%d)",
                         ERF_LEN / 1048576.0, ERF_GAIN * 100, ERF_NE);
            } else {
                mf = fopen(man, "a"); if (!mf) die("账本写不开");
                fprintf(mf, "%d -1 -1\n", L); fclose(mf);
                snprintf(status, sizeof status, "组合增益 %.1f%% ≤闸%.1f%% → 本层不注入(闸)", comb * 100, GATE * 100);
            }
        } else {
            if (ERF_ADD) {          /* ★叠加★: z/GE 之上再追加 ERF 残差补丁 */
                add = (uint8_t *)realloc(add, add_len + ERF_LEN); if (!add) die("realloc");
                memcpy(add + add_len, ERF_ADD, ERF_LEN); add_len += ERF_LEN; nrec_add++;
            }
            struct stat ds;
            if (stat(dql, &ds)) die("dql 不存在: %s", dql);
            long long osz = (long long)ds.st_size;
            FILE *df = fopen(dql, "r+b");
            if (!df) die("dql 打不开(r+b): %s", dql);
            uint32_t n0;
            if (fseeko(df, 8, SEEK_SET) || fread(&n0, 4, 1, df) != 1) die("dql nrec 读不到");
            if (fseeko(df, 0, SEEK_END)) die("dql seek end 失败");
            if (add_len && fwrite(add, 1, add_len, df) != add_len) die("dql 追加失败");
            uint32_t n1 = n0 + (uint32_t)nrec_add;
            if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
            fclose(df);
            mf = fopen(man, "a"); if (!mf) die("账本写不开");
            fprintf(mf, "%d %lld %u\n", L, osz, n0); fclose(mf);
            snprintf(status, sizeof status, "注入完(+%d记录, 原长 %lld 入账本)", nrec_add, osz);
        }
        flock(lkfd, LOCK_UN); close(lkfd);
    }
    double t3 = now_s();

    double gemean;
    { /* ★numpy 对 f16 数组的 mean: f32 累加/相除, 但【结果再舍回 f16】★
       * (numpy _methods._mean 结尾的 `if is_float16_result: ret = arr.dtype.type(ret)`)
       * 一期漏了最后这一步。夹具金标里撞出来: 只有 3 个专家被路由到时,
       * py 打 GE均值 1.0010 而 C 打 1.0005 —— 差的正是 f16 在 1.0 附近的一格(2⁻¹⁰)。
       * 一期是在 256 个专家全被路由到的 XCAP 层上对的拍, 那时两者恰好落同一格所以没露。
       * 只影响这一行打印, 不碰任何载荷字节。
       * (f32 累加这里是顺序累加, numpy 是 pairwise; 差在 f16 舍入前就被吸收掉了。) */
      float acc = 0.0f;
      for (int e = 0; e < NEXP; e++) acc += f16_to_f32(ge16[e]);
      gemean = (double)f16_to_f32(f64_to_f16((double)(acc / (float)NEXP))); }
    /* ★体积用自适应单位(2026-08-29 用户指出"GE 应该也有体积")★
     * 原来固定 %.1fMB: z 判空、只落 GE 的层打出来是"体积 0.0MB", 看着像什么都没落 ——
     * 实际 GE 记录 = 116B 头 + NEXP(256) × f16 = 628B, 43 层合计 26.4KB。
     * 打印精度不该把真实存在的产物显示成 0。 */
    char volbuf[32];
    if (add_len >= 1048576)   snprintf(volbuf, sizeof volbuf, "%.1fMB", add_len / 1048576.0);
    else if (add_len >= 1024) snprintf(volbuf, sizeof volbuf, "%.1fKB", add_len / 1024.0);
    else                      snprintf(volbuf, sizeof volbuf, "%zuB", add_len);
    printf("★L%d z侧车: held挽回 z^L %.1f%% 组合 %.1f%%  GE均值 %.4f  体积 %s | "
           "缓存 %.0fs 解算 %.0fs 总 %.0fs | %s\n",
           L, rz * 100, comb * 100, gemean, volbuf,
           t1 - t0, t2 - t1, t3 - t0, status);
#ifdef ZL_CUDA
    fprintf(stderr, "[zg] gemm卸载 命中=%ld 回落=%ld\n", zg_hits, zg_miss);
#endif
    return 0;
}
