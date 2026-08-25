    /* GGUF 标量模式: 学生权重来源换成 GGUF 专家张量切片, 不再读 dql_vq blob */
    gg_ctx gg; memset(&gg, 0, sizeof gg);
    if (GGP) gg_open(&gg, GGP);

    if (PREV && !XCAP)   /* py 里 PREV 整块嵌在 if XCAP: 内 —— 没 XCAP 时它是死的 */
        printf("  L%d 提示: 给了 PREV 目录但没给 XCAP 目录 — .py 里 PREV 只在 XCAP 口径下生效, 本次忽略\n", L);
    if (XCAP && PREV) {
        /* 第二轮反修: y_未修 = raw_ffn_out(带) / (1 + g_旧(新输入)) */
        snprintf(path, sizeof path, "%s/zrec_L%02d.bin", PREV, L);
        FILE *pf = fopen(path, "rb");
        if (!pf) die("上一轮记录打不开: %s", path);
        uint8_t hdr[116];
        if (fread(hdr, 1, 116, pf) != 116) die("%s 头截断", path);
        char nm[17]; memcpy(nm, hdr, 16); nm[16] = 0;
        if (!strstr(nm, "zl.AMPD")) die("assert 失败: L%d 上一轮记录不是 AMPD(%s), 无法还原增益", L, nm);
        uint64_t psz; memcpy(&psz, hdr + 88, 8);
        uint8_t *pay = (uint8_t *)xmalloc((size_t)psz);
        if (fread(pay, 1, (size_t)psz, pf) != psz) die("%s 载荷截断", path);
        fclose(pf);
        uint32_t pk, pdi, pdo; float psc;
        memcpy(&pk, pay, 4); memcpy(&psc, pay + 4, 4); memcpy(&pdi, pay + 8, 4); memcpy(&pdo, pay + 12, 4);
        if ((int)pdi != D) die("上一轮 din=%u ≠ %d — .py 的 X0@_V 在 din=3D 时也会崩, 口径不明拒跑", pdi, D);
        const uint16_t *h = (const uint16_t *)(pay + 16);
        size_t nA = (size_t)pdi * pk, nU = (size_t)pdo * pk;
        float *A = (float *)xmalloc(nA * 4), *U = (float *)xmalloc(nU * 4), *V = (float *)xmalloc(nA * 4);
        for (size_t i = 0; i < nA; i++) A[i] = f16_to_f32(h[i]);
        for (size_t i = 0; i < nU; i++) U[i] = f16_to_f32(h[nA + i]);
        for (size_t i = 0; i < nA; i++) V[i] = f16_to_f32(h[nA + nU + i]);
        float *pv = (float *)xmalloc((size_t)NTOK * pk * 4);
        float *pa = (float *)xmalloc((size_t)NTOK * pk * 4);
        mm32(0, 0, NTOK, (int)pk, D, X0, D, V, (int)pk, pv, (int)pk);
        mm32(0, 0, NTOK, (int)pk, D, X0, D, A, (int)pk, pa, (int)pk);
        for (size_t i = 0; i < (size_t)NTOK * pk; i++) pv[i] = tanhf(pv[i] / psc) * tanhf(pa[i] / psc);
        float *g = (float *)xmalloc((size_t)NTOK * D * 4);
        mm32(0, 1, NTOK, D, (int)pk, pv, (int)pk, U, (int)pk, g, D);
        double *ag = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
        for (size_t i = 0; i < (size_t)NTOK * D; i++) {
            float den = 1.0f + g[i];
            /* py: |den|<1e-3 时换成 sign(den)*1e-3 —— 注意 np.sign(0)=0, 换出来还是 0
             * (随后除零得 inf)。照抄, 不加"改进"。 */
            if (fabsf(den) < 1e-3f) den = (den > 0 ? 1e-3f : (den < 0 ? -1e-3f : 0.0f));
            YQE[i] = YQE[i] / den;
            ag[i] = fabs((double)g[i]);
        }
        qsort(ag, (size_t)NTOK * D, sizeof(double), cmp_dbl);
        size_t ng = (size_t)NTOK * D;
        double med = (ng % 2) ? ag[ng / 2] : 0.5 * (ag[ng / 2 - 1] + ag[ng / 2]);
        double pos = 0.99 * (double)(ng - 1);
        size_t lo = (size_t)pos; double fr = pos - (double)lo;
        double p99 = ag[lo] + fr * (ag[lo + 1 < ng ? lo + 1 : lo] - ag[lo]);
        printf("  L%d 第二轮: 已除回上一轮增益 |g_旧| 中位 %.5f p99 %.5f\n", L, med, p99);
        free(A); free(U); free(V); free(pv); free(pa); free(g); free(ag); free(pay);
    }

    /* ★对齐自检★ 判的是"两种对齐哪个对"的离散问题, 不是拿绝对余弦当质量闸:
     * 绝对值随层数衰减(L0 0.89 → L42 0.56), 拿常数当门会把深层全误拦。
     * 非 XCAP 时 X0 就是 X0fp 自己, 自检恒等于 1 —— py 也只在 XCAP 分支里做, 这里同。 */
    if (XCAP) {
        double c_ok = cosmed(X0fp, X0, NTOK, D);
        double c_bad = cosmed(X0fp + (size_t)D, X0, NTOK - 1, D);
        if (!(c_ok > c_bad * 1.15))
            die("assert 失败: L%d XCAP 对齐自检失败: 采用对齐 %.4f 未明显优于错位版 %.4f — 口径可疑, 停车",
                L, c_ok, c_bad);
        printf("  L%d XCAP: 对齐 %.4f vs 错位 %.4f (%.1f×) |x|=%.1f |y_q|=%.1f\n",
               L, c_ok, c_bad, c_ok / (c_bad > 1e-6 ? c_bad : 1e-6),
               fro_norm(X0, (size_t)NTOK * D), fro_norm(YQE, (size_t)NTOK * D));
    }

    /* ---------------- 缓存: dH 与配对记录 ---------------- */
    const long long NPAIR = (long long)NTOK * NACT;
    float *dH = NULL, *pw = NULL, *pYQ = NULL;
    int *prow = NULL, *pe = NULL;
    long long npair = 0;
    char cache[1200];
    snprintf(cache, sizeof cache, "%s/zcache_L%02d.npz", ld, L);

    struct stat cst;
    if (!stat(cache, &cst) && cst.st_size > 0) {
        int fd = open(cache, O_RDONLY);
        if (fd < 0) die("zcache 打不开: %s", cache);
        size_t zsz = (size_t)cst.st_size;
        uint8_t *zb = (uint8_t *)mmap(NULL, zsz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (zb == MAP_FAILED) die("zcache mmap 失败");
        npy_t adH, aprow, ape, apw, apYQ;
        if (npz_find(zb, (long long)zsz, "dH", &adH) || npz_find(zb, (long long)zsz, "prow", &aprow) ||
            npz_find(zb, (long long)zsz, "pe", &ape) || npz_find(zb, (long long)zsz, "pw", &apw) ||
            npz_find(zb, (long long)zsz, "pYQ", &apYQ)) die("zcache 字段缺: %s", cache);
        if (adH.d0 != NTOK || adH.d1 != D) die("zcache dH 形状 %lldx%lld ≠ %dx%d", adH.d0, adH.d1, NTOK, D);
        npair = aprow.d0;
        dH = (float *)xmalloc((size_t)NTOK * D * 4);          npz_to_f32(&adH, dH, (long long)NTOK * D);
        prow = (int *)xmalloc((size_t)npair * sizeof(int));   npz_to_i32(&aprow, prow, npair);
        pe = (int *)xmalloc((size_t)npair * sizeof(int));     npz_to_i32(&ape, pe, npair);
        pw = (float *)xmalloc((size_t)npair * 4);             npz_to_f32(&apw, pw, npair);
        pYQ = (float *)xmalloc((size_t)npair * D * 4);        npz_to_f32(&apYQ, pYQ, npair * D);
        for (long long i = 0; i < npair; i++)                 /* 缓存是外部产物, 索引越界会写飞内存 */
            if (prow[i] < 0 || prow[i] >= NTOK || pe[i] < 0 || pe[i] >= NEXP)
                die("zcache 配对越界: prow=%d pe=%d (NTOK=%d)", prow[i], pe[i], NTOK);
        munmap(zb, zsz); close(fd);
    } else {
        st_ctx *sc = (st_ctx *)xmalloc(sizeof(st_ctx));
        st_open(sc, hf);
        /* GGUF 标量模式下 py 是 `blob=None if _GG else open(...)` —— 根本不碰 dql_vq。
         * 所以这里也只在非 GGUF 时才要求侧车存在(全q2 底座那条产线压根没有 dql_vq)。 */
        const uint8_t *blob = NULL; size_t bsz = 0; int bfd = -1;
        if (!GGP) {
            snprintf(path, sizeof path, "%s/dql_vq_L%02d.bin", ld, L);
            bfd = open(path, O_RDONLY);
            if (bfd < 0) die("VQ 侧车打不开: %s", path);
            struct stat bst; fstat(bfd, &bst);
            bsz = (size_t)bst.st_size;
            blob = (const uint8_t *)mmap(NULL, bsz, PROT_READ, MAP_PRIVATE, bfd, 0);
            if (blob == MAP_FAILED) die("VQ 侧车 mmap 失败: %s", path);
        }

        /* need = 教师路由用到的专家 ∪ 部署路由用到的专家(py: XAP 时取并集) */
        int seen[NEXP]; memset(seen, 0, sizeof seen);
        for (long long i = 0; i < (long long)NTOK * NACT; i++) {
            int e = ridx[i];
            if (e < 0 || e >= NEXP) die("锚 ridx 越界: %d", e);
            seen[e] = 1;
        }
        if (XAP) for (long long i = 0; i < (long long)NTOK * NACT; i++) {
            int e = ridxq[i];
            if (e < 0 || e >= NEXP) die("链态锚 ridx 越界: %d", e);
            seen[e] = 1;
        }
        int need[NEXP], nneed = 0;
        for (int e = 0; e < NEXP; e++) if (seen[e]) need[nneed++] = e;

        dH = (float *)xcalloc((size_t)NTOK * D, 4);
        prow = (int *)xmalloc((size_t)NPAIR * sizeof(int));
        pe = (int *)xmalloc((size_t)NPAIR * sizeof(int));
        pw = (float *)xmalloc((size_t)NPAIR * 4);
        pYQ = (float *)xmalloc((size_t)NPAIR * D * 4);

        int *rows = (int *)xmalloc((size_t)NTOK * NACT * sizeof(int));
        int *slots = (int *)xmalloc((size_t)NTOK * NACT * sizeof(int));
        int *rowsq = XAP ? (int *)xmalloc((size_t)NTOK * NACT * sizeof(int)) : NULL;
        int *slotsq = XAP ? (int *)xmalloc((size_t)NTOK * NACT * sizeof(int)) : NULL;
        float *xs = (float *)xmalloc((size_t)NTOK * D * 4);
        float *Y = (float *)xmalloc((size_t)NTOK * D * 4);
        float *gb = NULL, *ub = NULL;
        int MOEI = 0;

        for (int i = 0; i < nneed; i++) {
            int e = need[i];
            int nr = 0;
            for (int t = 0; t < NTOK; t++)                    /* np.where(ridx==e): 行升序, 行内槽升序 */
                for (int s = 0; s < NACT; s++)
                    if (ridx[(size_t)t * NACT + s] == e) { rows[nr] = t; slots[nr] = s; nr++; }
            int nrq = nr;
            if (XAP) {
                nrq = 0;
                for (int t = 0; t < NTOK; t++)
                    for (int s = 0; s < NACT; s++)
                        if (ridxq[(size_t)t * NACT + s] == e) { rowsq[nrq] = t; slotsq[nrq] = s; nrq++; }
            }
            if (!nr && !nrq) continue;                        /* py: len(rows)==0 and len(rowsq)==0 */
            const int *QR_ROW = XAP ? rowsq : rows, *QR_SLOT = XAP ? slotsq : slots;
            const float *XQSRC = XAP ? XQ0 : X0, *RWQ = XAP ? rwq : rw;
            float *Wf[3], *Wq[3];
            static const char *NMS[3] = {"w1", "w3", "w2"};
            for (int wi = 0; wi < 3; wi++) {
                char tn[256];
                snprintf(tn, sizeof tn, "layers.%d.ffn.experts.%d.%s.weight", L, e, NMS[wi]);
                long R = 0, C = 0;
                float *wf = st_read_weight(sc, tn, &R, &C);
                if (!wf) die("HF 权重读不到: %s", tn);
                long rq = 0, cq = 0;
                float *wq = GGP ? gg_expert(&gg, L, NMS[wi], e, &rq, &cq)
                                : vq_dequant(blob, bsz, vq_slot(blob, bsz, e, wi), &rq, &cq);
                if (wq && (R != rq || C != cq)) {             /* py: Wf.shape != Wq.shape → Wf = Wf.T */
                    float *tr = (float *)xmalloc((size_t)R * C * 4);
                    for (long r = 0; r < R; r++) for (long c = 0; c < C; c++) tr[(size_t)c * R + r] = wf[(size_t)r * C + c];
                    free(wf); wf = tr; long t = R; R = C; C = t;
                }
                Wf[wi] = wf; Wq[wi] = wq ? wq : wf;           /* 无 vq 槽 ⇒ 量化侧退回 FP(py 同) */
                if (wi == 0) {
                    if (C != D) die("w1 列数 %ld ≠ D=%d(L%d e%d) — 方向判定失败", C, D, L, e);
                    if (!MOEI) { MOEI = (int)R;
                        gb = (float *)xmalloc((size_t)NTOK * MOEI * 4);
                        ub = (float *)xmalloc((size_t)NTOK * MOEI * 4); }
                    if (R != MOEI) die("w1 行数 %ld ≠ MOEI=%d", R, MOEI);
                } else if (wi == 2 && (R != D || C != MOEI))
                    die("w2 形状 %ldx%ld ≠ %dx%d", R, C, D, MOEI);
            }
            /* FP 目标侧: (XCAP 时 x=引擎真值, 否则 x=锚 fin) + FP 锚路由 + FP 权重 */
            if (nr > 0) {
                for (int j = 0; j < nr; j++) memcpy(xs + (size_t)j * D, X0 + (size_t)rows[j] * D, (size_t)D * 4);
                zl_expert_fwd(xs, nr, Wf[0], Wf[1], Wf[2], MOEI, SWLIM, Y, gb, ub);
                for (int j = 0; j < nr; j++) {
                    float w = rw[(size_t)rows[j] * NACT + slots[j]];
                    float *dst = dH + (size_t)rows[j] * D;
                    const float *y = Y + (size_t)j * D;
                    for (int d2 = 0; d2 < D; d2++) dst[d2] += w * y[d2];
                }
            }
            /* 部署侧: 链模式=链态 x_q + 部署路由 + 量化权重; XCAP 时 dH 不减 Yq(直接用引擎
             * raw_ffn_out), 但 pYQ/pw 仍要留给 GE。ADDON 时学生权重先乘既有 GE。 */
            if (nrq > 0) {
                for (int j = 0; j < nrq; j++) memcpy(xs + (size_t)j * D, XQSRC + (size_t)QR_ROW[j] * D, (size_t)D * 4);
                zl_expert_fwd(xs, nrq, Wq[0], Wq[1], Wq[2], MOEI, SWLIM, Y, gb, ub);
                for (int j = 0; j < nrq; j++) {
                    float wq2 = RWQ[(size_t)QR_ROW[j] * NACT + QR_SLOT[j]];
                    if (ge_old) wq2 *= ge_old[e];             /* py: wq = wq*ge_old[e] */
                    if (!XCAP) {
                        float *dst = dH + (size_t)QR_ROW[j] * D;
                        const float *y = Y + (size_t)j * D;
                        for (int d2 = 0; d2 < D; d2++) dst[d2] -= wq2 * y[d2];
                    }
                    prow[npair] = QR_ROW[j];
                    pe[npair] = e;
                    pw[npair] = wq2;
                    memcpy(pYQ + (size_t)npair * D, Y + (size_t)j * D, (size_t)D * 4);
                    npair++;
                }
            }
            for (int wi = 0; wi < 3; wi++) { if (Wq[wi] != Wf[wi]) free(Wq[wi]); free(Wf[wi]); }
            if ((i + 1) % 64 == 0) printf("  L%d 缓存 …%d/%d\n", L, i + 1, nneed);
        }
        /* ADDON: 既有 z 的出力从 dH 扣掉 → dH = 贪心修完之后的残差(全 f32, 与 py 同) */
        if (ADDON && zo_z) {
            const float *Xq = XAP ? XQ0 : X0;
            const float *Phi = Xq;
            float *phi_buf = NULL;
            if (zo_di == 3 * D) {
                phi_buf = (float *)xmalloc((size_t)NTOK * 3 * D * 4);
                phi32_ctx pc = {Xq, phi_buf, D};
                parallel_for(NTOK, phi32_worker, &pc);
                Phi = phi_buf;
            } else if (zo_di != D) die("ADDON: 既有 z 的 din=%d 既非 D 也非 3D — py 的 _Phi 会形状不合, 拒跑", zo_di);
            if (zo_do != D) die("ADDON: 既有 z 的 dout=%d ≠ D=%d", zo_do, D);
            float *pv = (float *)xmalloc((size_t)NTOK * zo_k0 * 4);
            mm32(0, 0, NTOK, zo_k0, zo_di, Phi, zo_di, zo_V, zo_k0, pv, zo_k0);
            for (size_t t = 0; t < (size_t)NTOK; t++)
                for (int c = 0; c < zo_k0; c++) pv[t * zo_k0 + c] *= zo_z[c];
            float *zout = (float *)xmalloc((size_t)NTOK * D * 4);
            mm32(0, 1, NTOK, D, zo_k0, pv, zo_k0, zo_U, zo_k0, zout, D);
            for (size_t j = 0; j < (size_t)NTOK * D; j++) dH[j] -= zout[j];
            free(pv); free(zout); free(phi_buf);
        }
        /* dH = Σw·Y_fp(锚教师) − 引擎真实量化 routed。非 XCAP 时这一减法已经逐专家做过了。 */
        if (XCAP) for (size_t i = 0; i < (size_t)NTOK * D; i++) dH[i] -= YQE[i];

        free(rows); free(slots); free(rowsq); free(slotsq); free(xs); free(Y); free(gb); free(ub);
        if (blob) { munmap((void *)blob, bsz); close(bfd); }

        /* 原子换名: 并行 amp_solve 只见完整 zcache */
        char tmp[1300]; snprintf(tmp, sizeof tmp, "%s.tmp.npz", cache);
        zwr_t zw; memset(&zw, 0, sizeof zw);
        zw.f = fopen(tmp, "wb");
        if (!zw.f) die("zcache 写不开: %s", tmp);
        int64_t *prow64 = (int64_t *)xmalloc((size_t)npair * 8);
        for (long long i = 0; i < npair; i++) prow64[i] = prow[i];
        zw_add(&zw, "dH", "<f4", NTOK, D, 2, dH, 4);
        zw_add(&zw, "prow", "<i8", npair, 1, 1, prow64, 8);
        zw_add(&zw, "pe", "<i4", npair, 1, 1, pe, 4);
        zw_add(&zw, "pw", "<f4", npair, 1, 1, pw, 4);
        zw_add(&zw, "pYQ", "<f4", npair, D, 2, pYQ, 4);
        zw_add(&zw, "pDY", "<f4", npair, D, 2, NULL, 4);      /* py 存的就是全零 */
        if (XCAP) {                                           /* py: **({"yqe":YQE,"xcap":X0} if XCAP else {}) */
            zw_add(&zw, "yqe", "<f4", NTOK, D, 2, YQE, 4);
            zw_add(&zw, "xcap", "<f4", NTOK, D, 2, X0, 4);
        }
        zw_finish(&zw);
        fclose(zw.f);
        free(prow64);
        if (rename(tmp, cache)) die("zcache 换名失败: %s", strerror(errno));
        free(sc);
    }
    double t1 = now_s();
    if (CACHE_ONLY) { printf("★L%d zcache-only: 就绪(%.0fs), 解算交外部\n", L, t1 - t0); return 0; }

    /* ---------------- 解算 ---------------- */
    /* py: X=(XQ0 if XAP else X0).astype(np.float64) —— 链模式下解算的自变量是【链态】x_q,
     * 因为放大器部署时看到的就是它; 非链模式下 X0 已经是 XCAP 真值或锚 fin。 */
    const float *XSOLVE = XAP ? XQ0 : X0;
    double *X = (double *)xmalloc((size_t)NTOK * D * sizeof(double));
    for (size_t i = 0; i < (size_t)NTOK * D; i++) X[i] = XSOLVE[i];

    int ntr = 0, nev = 0, *tr = NULL, *ev = NULL;
    const char *fr = getenv("DS4_ZL_FIT_RANGES"), *er = getenv("DS4_ZL_EV_RANGE");
    if (fr) {
        if (!er) die("DS4_ZL_FIT_RANGES 设了但 DS4_ZL_EV_RANGE 没设 — .py 同样会崩");
        tr = parse_ranges(fr, &ntr);
        ev = parse_ranges(er, &nev);
    } else {
        int NF = env_int("DS4_ZL_NFIT", 1287);
        ntr = NF; nev = NTOK - NF;
        tr = (int *)xmalloc((size_t)ntr * sizeof(int));
        ev = (int *)xmalloc((size_t)(nev > 0 ? nev : 1) * sizeof(int));
        for (int i = 0; i < ntr; i++) tr[i] = i;
        for (int i = 0; i < nev; i++) ev[i] = NF + i;
    }
    for (int i = 0; i < ntr; i++) if (tr[i] < 0 || tr[i] >= NTOK) die("fit 行号 %d 越界", tr[i]);
    for (int i = 0; i < nev; i++) if (ev[i] < 0 || ev[i] >= NTOK) die("ev 行号 %d 越界", ev[i]);

    /* colw = sqrt(var(dH[tr], axis=0) + 1e-12) —— ★是 f32★(py 里 dH 是 f32, .var(0) 不升精度),
     * 于是 Ra=dH[tr]*colw 也是 f32, 到 np.linalg.solve 才升 f64。这里照抄这个精度阶梯。
     * (均值/方差的求和用 f64 累加器, 与 numpy 的 f32 pairwise 差 ~1e-7 相对, 打印精度内。) */
    float *colw = (float *)xmalloc((size_t)D * 4);
    for (int j = 0; j < D; j++) {
        double m = 0;
        for (int i = 0; i < ntr; i++) m += dH[(size_t)tr[i] * D + j];
        m /= ntr;
        double v = 0;
        for (int i = 0; i < ntr; i++) { double d2 = dH[(size_t)tr[i] * D + j] - m; v += d2 * d2; }
        colw[j] = sqrtf((float)(v / ntr) + 1e-12f);
    }

    const int NN = 2 * ntr;
    double *Xa = (double *)xmalloc((size_t)NN * D * sizeof(double));
    float *Ra32 = (float *)xmalloc((size_t)NN * D * 4);
    {
        mt_t rng; mt_seed(&rng, 1);          /* np.random.RandomState(1) */
        for (int i = 0; i < ntr; i++) {
            const double *x = X + (size_t)tr[i] * D;
            memcpy(Xa + (size_t)i * D, x, (size_t)D * sizeof(double));
            double ss = 0;
            for (int j = 0; j < D; j++) ss += x[j] * x[j];
            double rms = sqrt(ss / D);
            double *xd = Xa + (size_t)(ntr + i) * D;
            /* randn 按 C 序逐元素抽 —— 行内先走完再下一行, 与 randn(len(tr),D) 一致 */
            for (int j = 0; j < D; j++) xd[j] = (x[j] + (mt_gauss(&rng) * 0.04) * rms) * 0.5;
            const float *dh = dH + (size_t)tr[i] * D;
            float *r0 = Ra32 + (size_t)i * D, *r1 = Ra32 + (size_t)(ntr + i) * D;
            for (int j = 0; j < D; j++) { r0[j] = dh[j] * colw[j]; r1[j] = r0[j] * 0.5f; }
        }
    }

    int kmax = 0;
    int KG[16], nKG = 0;
    {
        int cand[9] = {64, 128, 256, 384, 512, 768, 1024, 1536, K};
        for (int i = 0; i < 9; i++) {
            int k = cand[i];
            if (k > K || k > 1024) continue;
            int dup = 0;
            for (int j = 0; j < nKG; j++) if (KG[j] == k) dup = 1;
            if (!dup) KG[nKG++] = k;
        }
        for (int a = 0; a < nKG; a++) for (int b = a + 1; b < nKG; b++)
            if (KG[b] < KG[a]) { int t = KG[a]; KG[a] = KG[b]; KG[b] = t; }
        if (!nKG) { KG[nKG++] = K > 0 ? K : 0; }
        kmax = KG[nKG - 1];
    }

    /* --- 线性支线: 对偶 ridge(dither 增广 + colw 感知列权) → 折 colw → 低秩截断 --- */
    double *A = NULL, *S = NULL, *Bt = NULL, *Af = NULL, *Sf = NULL, *Bf = NULL;
    int rlin = 0, rfta = 0;
    {
        double *G = (double *)xmalloc((size_t)NN * NN * sizeof(double));
        mm64(0, 1, NN, NN, D, Xa, D, Xa, D, G, NN);
        double trc = 0;
        for (int i = 0; i < NN; i++) trc += G[(size_t)i * NN + i];
        double ridge = 3.0 * trc / D + 1e-10;
        for (int i = 0; i < NN; i++) G[(size_t)i * NN + i] += ridge;
        double *al = (double *)xmalloc((size_t)NN * D * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * D; i++) al[i] = Ra32[i];
        if (chol_solve(G, NN, al, D)) die("对偶 ridge Cholesky 失败(G 非正定) — 停车");
        free(G);
        double *Wz = (double *)xmalloc((size_t)D * D * sizeof(double));
        mm64(1, 0, D, D, NN, Xa, D, al, D, Wz, D);
        free(al);
        for (int i = 0; i < D; i++) { double *row = Wz + (size_t)i * D;
            for (int j = 0; j < D; j++) row[j] /= colw[j]; }
        if (kmax > 0) {
            rlin = kmax + 64; if (rlin > D) rlin = D;
            zl_svd_lowrank(Wz, D, D, rlin, &A, &S, &Bt);
        }
        free(Wz);
    }

    /* --- ftA 特征提升支线: φ(x)=[x, x⊙x/rms, relu(x)], 部署 din=3D --- */
    float *Phi_ev32 = NULL;
    const int DIN3 = 3 * D;
    if (FTA) {
        float *Fa = (float *)xmalloc((size_t)NN * DIN3 * 4);
        { phi64_ctx pc = {Xa, Fa, D}; parallel_for(NN, phi64_worker, &pc); }
        float *Gf32 = (float *)xmalloc((size_t)NN * NN * 4);
        mm32(0, 1, NN, NN, DIN3, Fa, DIN3, Fa, DIN3, Gf32, NN);     /* py: Fa 是 f32 → sgemm */
        double trc = 0;
        for (int i = 0; i < NN; i++) trc += Gf32[(size_t)i * NN + i];
        float rg = (float)(3.0 * (double)(float)trc / DIN3 + 1e-10);
        for (int i = 0; i < NN; i++) Gf32[(size_t)i * NN + i] += rg;
        double *Gf = (double *)xmalloc((size_t)NN * NN * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * NN; i++) Gf[i] = Gf32[i];
        free(Gf32);
        double *alf = (double *)xmalloc((size_t)NN * D * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * D; i++) alf[i] = Ra32[i];
        if (chol_solve(Gf, NN, alf, D)) die("ftA 支线 Cholesky 失败 — 停车");
        free(Gf);
        double *Fa64 = (double *)xmalloc((size_t)NN * DIN3 * sizeof(double));
        for (size_t i = 0; i < (size_t)NN * DIN3; i++) Fa64[i] = Fa[i];
        free(Fa);
        double *Wf = (double *)xmalloc((size_t)DIN3 * D * sizeof(double));
        mm64(1, 0, DIN3, D, NN, Fa64, DIN3, alf, D, Wf, D);
        free(Fa64); free(alf);
        for (int i = 0; i < DIN3; i++) { double *row = Wf + (size_t)i * D;
            for (int j = 0; j < D; j++) row[j] /= colw[j]; }
        if (kmax > 0) {
            rfta = kmax + 64; if (rfta > D) rfta = D;
            zl_svd_lowrank(Wf, DIN3, D, rfta, &Af, &Sf, &Bf);
        }
        free(Wf);
        Phi_ev32 = (float *)xmalloc((size_t)nev * DIN3 * 4);
        float *Xev32 = (float *)xmalloc((size_t)nev * D * 4);
        for (int i = 0; i < nev; i++) for (int j = 0; j < D; j++) Xev32[(size_t)i * D + j] = (float)X[(size_t)ev[i] * D + j];
        { phi32_ctx pc = {Xev32, Phi_ev32, D}; parallel_for(nev, phi32_worker, &pc); }
        free(Xev32);
    }
    free(Xa);

