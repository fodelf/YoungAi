/* zlayer_build.inc.c — 教师/学生专家配对构建(2026-08-31 语义抽取, 只被 zlayer.c include,
 * 位置在 p2 之后=文件作用域)。
 *
 * ★为什么抽出来★ 原是 p4 main 内联块(py 金标口径), 只服务主解算(反修锚全量行)。
 * 三等分语料落盘后跨语料闸(--gate-anchor)也要同一口径的 dH/配对 —— 抄第二份=
 * 反修一份实现铁律的死罪, 故抽成唯一函数, 主解算与闸两处同源。算术与循环序
 * 逐字未动(migrate/golden.txt 金标建立在这套字节序上, 动=重跑金标)。
 *
 * dH = Σw·Y_fp(教师) − Σwq·Y_q(学生, sub_student=0 时跳过减法=XCAP+YQE 口径,
 * 主调自行减引擎真值); 配对(prow/pe/pw/pYQ)恒建(GE 臂数据面)。全部输出内部分配。 */
static void zl_expert_fwd(const float *x, int n, const float *w1, const float *w3,
                          const float *w2, int MOEI, float lim, float *Y, float *g, float *u);

static void zl_build_pairs(const char *hf, const char *ld, const char *GGP, gg_ctx *gg,
                           int L, int NTOK, int NACT, float SWLIM,
                           const float *X0, const int *ridx, const float *rw,
                           int XAP, const float *XQ0, const int *ridxq, const float *rwq,
                           const float *ge_old, int sub_student,
                           float **dH_out, int **prow_out, int **pe_out,
                           float **pw_out, float **pYQ_out, long long *npair_out) {
    char path[1024];
    const long long NPAIR = (long long)NTOK * NACT;
    long long npair = 0;
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

    float *dH = (float *)xcalloc((size_t)NTOK * D, 4);
    int *prow = (int *)xmalloc((size_t)NPAIR * sizeof(int));
    int *pe = (int *)xmalloc((size_t)NPAIR * sizeof(int));
    float *pw = (float *)xmalloc((size_t)NPAIR * 4);
    float *pYQ = (float *)xmalloc((size_t)NPAIR * D * 4);

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
            float *wq = GGP ? gg_expert(gg, L, NMS[wi], e, &rq, &cq)
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
                if (sub_student) {   /* 只换 x 模式: 学生由本进程重算, 逐专家减 */
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
    free(rows); free(slots); free(rowsq); free(slotsq); free(xs); free(Y); free(gb); free(ub);
    if (blob) { munmap((void *)blob, bsz); close(bfd); }
    free(sc);
    *dH_out = dH; *prow_out = prow; *pe_out = pe;
    *pw_out = pw; *pYQ_out = pYQ; *npair_out = npair;
}
