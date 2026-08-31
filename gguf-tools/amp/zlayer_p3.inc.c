/* ---------------- 低秩 SVD: 子空间迭代 + Rayleigh-Ritz ----------------
 * ★与 numpy 的差别★ .py 走 np.linalg.svd(LAPACK gesdd, 精确全谱)。这里只求前 r 个奇异
 * 三元组: 随机起始 → (Wᵀ 再 W) 幂迭代 12 轮(与 ds4quant_run.c z_solve_dual 尾部同款) →
 * 用 B=VᵀW 的小核 BBᵀ(r×r) 精确特征分解回收奇异值/向量(Rayleigh-Ritz)。
 * 精确性: 只要 V 张成的子空间收敛到前 r 个左奇异方向, 结果就等于精确截断; 收敛速度取决
 * 于 σ_r/σ_{r+p} 的间隙, 所以过采样 p=64。【故不可与 .py 逐位对拍】, 判定看 held 挽回率。
 * 出参: A[din*r](左, 列存 c), S[r](降序), Bt[r*Dout](右, 行 c)。 */
typedef struct { double *V; int rows, r; const double *G; int gcols; } cgs_ctx;

/* 块 Gram-Schmidt 正交化(带一次重正交): V[rows,r] 就地正交归一 */
static void orthonormalize(double *V, int rows, int r) {
    const int BS = 64;
    double *C = (double *)xmalloc((size_t)r * BS * sizeof(double));
    double *T = (double *)xmalloc((size_t)rows * BS * sizeof(double));
    for (int j0 = 0; j0 < r; j0 += BS) {
        int bs = j0 + BS > r ? r - j0 : BS;
        for (int pass = 0; pass < 2 && j0 > 0; pass++) {
            /* C[j0,bs] = Vprevᵀ · Vblk ; Vblk -= Vprev·C */
            mm64(1, 0, j0, bs, rows, V, r, V + j0, r, C, bs);
            mm64(0, 0, rows, bs, j0, V, r, C, bs, T, bs);
            for (int i = 0; i < rows; i++) {
                double *v = V + (size_t)i * r + j0;
                const double *t = T + (size_t)i * bs;
                for (int j = 0; j < bs; j++) v[j] -= t[j];
            }
        }
        for (int j = 0; j < bs; j++) {                  /* 块内 MGS */
            int col = j0 + j;
            for (int rep = 0; rep < 2; rep++) {
                for (int k = j0; k < col; k++) {
                    double d = 0;
                    for (int i = 0; i < rows; i++) d += V[(size_t)i * r + k] * V[(size_t)i * r + col];
                    for (int i = 0; i < rows; i++) V[(size_t)i * r + col] -= d * V[(size_t)i * r + k];
                }
            }
            double nr = 0;
            for (int i = 0; i < rows; i++) nr += V[(size_t)i * r + col] * V[(size_t)i * r + col];
            nr = sqrt(nr);
            if (nr < 1e-200) { for (int i = 0; i < rows; i++) V[(size_t)i * r + col] = (i == col % rows) ? 1.0 : 0.0; nr = 1.0; }
            double inv = 1.0 / nr;
            for (int i = 0; i < rows; i++) V[(size_t)i * r + col] *= inv;
        }
    }
    free(C); free(T);
}

static void zl_svd_lowrank(const double *W, int din, int dout, int r,
                           double **A_out, double **S_out, double **Bt_out) {
    if (r > din) r = din;
    if (r > dout) r = dout;
    double *V = (double *)xmalloc((size_t)din * r * sizeof(double));
    double *T = (double *)xmalloc((size_t)dout * r * sizeof(double));
    uint64_t seed = 0x5A5A1EEDULL;                       /* 固定种子: 同输入必同产物 */
    for (size_t i = 0; i < (size_t)din * r; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        V[i] = (double)((seed >> 40) & 0xFFFFFF) / 16777216.0 - 0.5;
    }
    orthonormalize(V, din, r);
    for (int it = 0; it < 12; it++) {
        mm64(1, 0, dout, r, din, W, dout, V, r, T, r);   /* T = Wᵀ V */
        mm64(0, 0, din, r, dout, W, dout, T, r, V, r);   /* V = W  T */
        orthonormalize(V, din, r);
    }
    double *B = (double *)xmalloc((size_t)r * dout * sizeof(double));
    mm64(1, 0, r, dout, din, V, r, W, dout, B, dout);    /* B = Vᵀ W  [r,dout] */
    double *C = (double *)xmalloc((size_t)r * r * sizeof(double));
    mm64(0, 1, r, r, dout, B, dout, B, dout, C, r);      /* C = B Bᵀ  [r,r] */
    double *lam = (double *)xmalloc((size_t)r * sizeof(double));
    double *E = (double *)xmalloc((size_t)r * r * sizeof(double));
    jacobi_eig(C, r, lam, E);
    free(C);
    double *S = (double *)xmalloc((size_t)r * sizeof(double));
    for (int c = 0; c < r; c++) S[c] = lam[c] > 0 ? sqrt(lam[c]) : 0.0;
    free(lam);
    double *A = (double *)xmalloc((size_t)din * r * sizeof(double));
    mm64(0, 0, din, r, r, V, r, E, r, A, r);             /* A = V E */
    double *Bt = (double *)xmalloc((size_t)r * dout * sizeof(double));
    mm64(1, 0, r, dout, r, E, r, B, dout, Bt, dout);     /* Bt = Eᵀ B, 下面按 1/S 归一 */
    double smax = S[0];
    for (int c = 0; c < r; c++) {
        double inv = (S[c] > 1e-13 * smax && S[c] > 0) ? 1.0 / S[c] : 0.0;
        if (inv == 0.0) S[c] = 0.0;
        double *row = Bt + (size_t)c * dout;
        for (int j = 0; j < dout; j++) row[j] *= inv;
    }
    free(V); free(T); free(B); free(E);
    *A_out = A; *S_out = S; *Bt_out = Bt;
}

/* ---------------- Householder QR(reduced) ----------------
 * 用在两处: ADDON 的新旧低秩合并(np.linalg.qr(Pc)/qr(Qc)) 与 ERF 的 randomized SVD。
 * ★与 numpy 不逐位★: np.linalg.qr 走 LAPACK geqrf/orgqr —— 分块顺序、Householder 的符号
 * 约定(LAPACK 让 R 对角可正可负)都与这里的教科书写法不同。两处调用都只把 QR 当【中间基】:
 *   ADDON 最终产物 = Pc·Qcᵀ 的截断 SVD, 数学上与用哪组正交基无关;
 *   ERF  最终产物 = 低秩补丁 U·Vᵀ 的乘积, 同理。
 * 所以只要 Q 正交、QR=A 成立就够, 金标看打印读数不看载荷字节。
 *
 * 就地分解 A[m,n](行主序, m≥n): 返回时
 *   R = 上三角(对角取 rdiag[k], 上方取 A[k*n+j], j>k);
 *   第 k 个 Householder 向量 v_k 存在 A 的第 k 列 i≥k 处, tau[k]=2/‖v_k‖²(v_k=0 时 tau=0)。 */
/* 尾列更新 A[k:m, k+1:n] -= tau·v·(vᵀ·A[k:m, k+1:n])。parallel_for 的分片是 0 基,
 * 所以 worker 里把 jj 重映射成 j = k+1+jj。 */
typedef struct { double *A; int m, n, k; double tau; } hhu_ctx;
static void hhu_worker(void *vc, int i0, int i1) {
    hhu_ctx *c = (hhu_ctx *)vc;
    const int m = c->m, n = c->n, k = c->k;
    for (int jj = i0; jj < i1; jj++) {
        int j = k + 1 + jj;
        double d = 0;
        for (int i = k; i < m; i++) d += c->A[(size_t)i * n + k] * c->A[(size_t)i * n + j];
        d *= c->tau;
        if (d == 0.0) continue;
        for (int i = k; i < m; i++) c->A[(size_t)i * n + j] -= c->A[(size_t)i * n + k] * d;
    }
}
static void hh_qr(double *A, int m, int n, double *tau, double *rdiag) {
    if (m < n) die("hh_qr: m=%d < n=%d(本实现只做 m≥n 的 reduced QR)", m, n);
    for (int k = 0; k < n; k++) {
        double nrm = 0;
        for (int i = k; i < m; i++) { double x = A[(size_t)i * n + k]; nrm += x * x; }
        nrm = sqrt(nrm);
        if (nrm == 0.0) { tau[k] = 0.0; rdiag[k] = 0.0; continue; }
        double x0 = A[(size_t)k * n + k];
        double alpha = (x0 >= 0.0) ? -nrm : nrm;         /* 取反号避免相消 */
        A[(size_t)k * n + k] = x0 - alpha;
        double v2 = 0;
        for (int i = k; i < m; i++) { double v = A[(size_t)i * n + k]; v2 += v * v; }
        tau[k] = (v2 > 0.0) ? 2.0 / v2 : 0.0;
        rdiag[k] = alpha;
        if (k + 1 < n) { hhu_ctx hc = { A, m, n, k, tau[k] }; parallel_for(n - k - 1, hhu_worker, &hc); }
    }
}
/* Y[m,ncy] ← Q·Y, Q = H_0·H_1·…·H_{n-1}(只用前 n 个反射子)。
 * 不显式生成 Q: 调用点要的都是 Q 乘一个窄矩阵, 这样省掉 m×n 的中间物。 */
typedef struct { const double *A; int m, n, k, ncy; double *Y; double tau; } hha_ctx;
static void hha_worker(void *vc, int c0, int c1) {
    hha_ctx *c = (hha_ctx *)vc;
    const int m = c->m, n = c->n, k = c->k, ncy = c->ncy;
    for (int col = c0; col < c1; col++) {
        double d = 0;
        for (int i = k; i < m; i++) d += c->A[(size_t)i * n + k] * c->Y[(size_t)i * ncy + col];
        d *= c->tau;
        if (d == 0.0) continue;
        for (int i = k; i < m; i++) c->Y[(size_t)i * ncy + col] -= c->A[(size_t)i * n + k] * d;
    }
}
static void hh_apply_q(const double *A, const double *tau, int m, int n, double *Y, int ncy) {
    for (int k = n - 1; k >= 0; k--) {
        if (tau[k] == 0.0) continue;
        hha_ctx ac = { A, m, n, k, ncy, Y, tau[k] };
        parallel_for(ncy, hha_worker, &ac);
    }
}

/* 稳定降序 argsort(键相同按下标升序)。py 用 np.argsort(-_en) = 不稳定 quicksort;
 * _en 是连续浮点能量, 实测不撞值 —— 真撞了两边的 tau 也一样(同值), 只是记录里的
 * 专家内 token 门限来源不同, 不影响任何打印读数。 */
typedef struct { double v; int i; } dsort_t;
static int cmp_dsort_desc(const void *a, const void *b) {
    const dsort_t *x = (const dsort_t *)a, *y = (const dsort_t *)b;
    if (x->v > y->v) return -1;
    if (x->v < y->v) return 1;
    return x->i < y->i ? -1 : (x->i > y->i);
}

/* ---------------- 单专家前向(FP 或量化侧共用) ----------------
 * py: Y = swiglu(x@w1ᵀ, x@w3ᵀ) @ w2ᵀ, 路由权重【不在这里乘】(外面 dH[rows]+=w*Y)。
 * w1/w3 是 [MOEI, D], w2 是 [D, MOEI]; x[n,D] → Y[n,D]。全 f32(与 numpy 同)。 */
static void zl_expert_fwd(const float *x, int n, const float *w1, const float *w3,
                          const float *w2, int MOEI, float lim, float *Y, float *g, float *u) {
    mm32(0, 1, n, MOEI, D, x, D, w1, D, g, MOEI);
    mm32(0, 1, n, MOEI, D, x, D, w3, D, u, MOEI);
    for (size_t i = 0; i < (size_t)n * MOEI; i++) g[i] = zl_swiglu1(g[i], u[i], lim);
    mm32(0, 1, n, D, MOEI, g, MOEI, w2, MOEI, Y, D);
}

/* ---------------- XCAP 捕获读取与对齐自检 ----------------
 * 引擎走 DS4_EVAL_IDS 时流首插了 BOS, 捕获会比锚多一行(NTOK+1)。差这一行就是整体错位
 * 一格 —— 每个 token 的 x 配到前一个 token 的目标上, 不报错、照样出挽回率。 */
static float *capload(const char *xcap, const char *nm, int L, int NTOK, int cols) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s_L%d", xcap, nm, L);
    FILE *f = fopen(p, "rb");
    if (!f) die("捕获打不开: %s", p);
    fseeko(f, 0, SEEK_END);
    long long sz = ftello(f);
    long long n = sz / 2 / cols;
    if (n != NTOK && n != NTOK + 1)
        die("assert 失败: %s: %lld 行, 既非 NTOK=%d 也非 NTOK+1(BOS) — 口径不明, 拒跑", p, n, NTOK);
    long long off = n - NTOK;                            /* 1 = 掐掉流首 BOS 行 */
    if (fseeko(f, (off_t)(off * cols * 2), SEEK_SET)) die("%s seek 失败", p);
    uint16_t *h = (uint16_t *)xmalloc((size_t)NTOK * cols * 2);
    if (fread(h, 2, (size_t)NTOK * cols, f) != (size_t)NTOK * cols) die("%s 读不满", p);
    fclose(f);
    float *o = (float *)xmalloc((size_t)NTOK * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)NTOK * cols; i++) o[i] = f16_to_f32(h[i]);
    free(h);
    return o;
}

/* 逐行余弦的中位数(a,b 各取前 min(len) 行)。py 用 f32 算、np.median 对偶数长度取中间两个
 * 的均值 —— 这里用 f64 累加, 差在 1e-7 量级, 判的是"对齐 vs 错位"的离散问题, 不影响判据。 */
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static double cosmed(const float *a, const float *b, int n, int cols) {
    double *v = (double *)xmalloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        const float *x = a + (size_t)i * cols, *y = b + (size_t)i * cols;
        double d = 0, nx = 0, ny = 0;
        for (int j = 0; j < cols; j++) { d += (double)x[j] * y[j]; nx += (double)x[j] * x[j]; ny += (double)y[j] * y[j]; }
        v[i] = d / (sqrt(nx) * sqrt(ny) + 1e-9);
    }
    qsort(v, (size_t)n, sizeof(double), cmp_dbl);
    double m = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    free(v);
    return m;
}
static double fro_norm(const float *a, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (double)a[i] * a[i];
    return sqrt(s);
}

/* ---------------- 116B 记录头(py 的 rec()) ---------------- */
static uint8_t *make_rec(const char *nm, const void *pay, size_t psz, size_t *out_len) {
    uint8_t *r = (uint8_t *)xcalloc(DS4_AMP_REC_HDR + psz, 1);
    memcpy(r, nm, strlen(nm));
    uint64_t p64 = psz; memcpy(r + DS4_AMP_REC_OFF_PSZ, &p64, 8);
    int32_t one = 1; memcpy(r + DS4_AMP_REC_OFF_VD, &one, 4);
    if (psz) memcpy(r + DS4_AMP_REC_HDR, pay, psz);
    *out_len = DS4_AMP_REC_HDR + psz;
    return r;
}

/* ---------------- 区间解析: "a:b,c:d" → 行号数组 ---------------- */
static int *parse_ranges(const char *s, int *n_out) {
    int cap = 64, n = 0;
    int *v = (int *)xmalloc((size_t)cap * sizeof(int));
    const char *p = s;
    while (*p) {
        char *e1, *e2;
        long a = strtol(p, &e1, 10);
        if (*e1 != ':') die("区间语法错(要 a:b): %s", s);
        long b = strtol(e1 + 1, &e2, 10);
        for (long i = a; i < b; i++) {
            if (n == cap) { cap *= 2; v = (int *)realloc(v, (size_t)cap * sizeof(int)); if (!v) die("realloc"); }
            v[n++] = (int)i;
        }
        p = e2;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p && !(*p >= '0' && *p <= '9')) die("区间语法错: %s", s);
    }
    *n_out = n;
    return v;
}

/* ======================================================================== *
 *                                  main                                    *
 * ======================================================================== */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);          /* py 全程 flush=True: 战役脚本靠管道实时 grep */
    crc_init();

    if (argc > 1 && !strcmp(argv[1], "--selftest-rng")) {
        /* 对拍: python3 -c "import numpy as np; print(np.random.RandomState(1).randn(8))"
         * 期望 1.62434536 -0.61175641 -0.52817175 -1.07296862 0.86540763 -2.3015387
         *      1.74481176 -0.7612069 */
        mt_t s; mt_seed(&s, 1);
        for (int i = 0; i < 8; i++) printf("%.17g\n", mt_gauss(&s));
        return 0;
    }
    if (argc > 3 && !strcmp(argv[1], "--selftest-deq")) {
        /* GGUF 标量 dequant 与 gguf-py 对拍用的管道: stdin 吃 nblk 个块的原始字节,
         * stdout 吐 nblk*blk 个 f32。驱动见 migrate/zlayer_transcription_notes.md 的金标节。
         *   ./zlayer --selftest-deq <ggml类型号> <块数> < blocks.bin > out.f32 */
        uint32_t ty = (uint32_t)atoi(argv[2]);
        long nb = atol(argv[3]);
        uint64_t blk, tsz; gg_type_geom(ty, &blk, &tsz);
        size_t insz = (size_t)nb * tsz, outn = (size_t)nb * blk;
        uint8_t *in = (uint8_t *)xmalloc(insz);
        if (fread(in, 1, insz, stdin) != insz) die("--selftest-deq: stdin 只给了不足 %zu B", insz);
        float *out = (float *)xmalloc(outn * 4);
        gg_dequant(ty, in, outn, out);
        if (fwrite(out, 4, outn, stdout) != outn) die("--selftest-deq: stdout 写失败");
        return 0;
    }
    if (argc > 3 && !strcmp(argv[1], "--selftest-qr")) {
        /* Householder QR 自检: 随机 m×n, 报 ‖QR−A‖∞ 与 ‖QᵀQ−I‖∞(都应 ~1e-13)。
         * 与 numpy 不逐位是设计内的(见 hh_qr 注释), 这里判的是"分解本身对不对"。 */
        int m = atoi(argv[2]), n = atoi(argv[3]);
        if (m < n || n < 1) die("--selftest-qr: 需要 m≥n≥1");
        double *A0 = (double *)xmalloc((size_t)m * n * sizeof(double));
        mt_t s; mt_seed(&s, 3);
        for (size_t i = 0; i < (size_t)m * n; i++) A0[i] = mt_gauss(&s);
        double *A = (double *)xmalloc((size_t)m * n * sizeof(double));
        memcpy(A, A0, (size_t)m * n * sizeof(double));
        double *tau = (double *)xmalloc((size_t)n * sizeof(double));
        double *rd = (double *)xmalloc((size_t)n * sizeof(double));
        hh_qr(A, m, n, tau, rd);
        double *R = (double *)xcalloc((size_t)n * n, sizeof(double));
        for (int i = 0; i < n; i++) { R[(size_t)i * n + i] = rd[i];
            for (int j = i + 1; j < n; j++) R[(size_t)i * n + j] = A[(size_t)i * n + j]; }
        double *Q = (double *)xcalloc((size_t)m * n, sizeof(double));
        for (int c = 0; c < n; c++) Q[(size_t)c * n + c] = 1.0;
        hh_apply_q(A, tau, m, n, Q, n);
        double *QR = (double *)xmalloc((size_t)m * n * sizeof(double));
        mm64(0, 0, m, n, n, Q, n, R, n, QR, n);
        double e1 = 0;
        for (size_t i = 0; i < (size_t)m * n; i++) { double d2 = fabs(QR[i] - A0[i]); if (d2 > e1) e1 = d2; }
        double *QtQ = (double *)xmalloc((size_t)n * n * sizeof(double));
        mm64(1, 0, n, n, m, Q, n, Q, n, QtQ, n);
        double e2 = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double d2 = fabs(QtQ[(size_t)i * n + j] - (i == j ? 1.0 : 0.0)); if (d2 > e2) e2 = d2; }
        printf("QR自检 m=%d n=%d: ‖QR−A‖inf=%.3e ‖QᵀQ−I‖inf=%.3e\n", m, n, e1, e2);
        return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "用法: zlayer <hf> <layers_dir> <anchor> <L> <K(秩,必传)> [inject=1] [XCAP目录] [PREV目录]\n"
                        "             [--gguf P] [--xanchor P] [--addon] [--cache-only]\n"
                        "             --ntok N(必传) [--swlim F=10] [--fta N=1] [--ge N=1]\n"
                        "             [--gate F=0] [--ge-lam F=1e-3] [--erf N=1] [--erf-bar F=0.01] [--erf-r N=8]\n"
                        "      zlayer --selftest-rng\n");
        return 1;
    }
    const char *hf = argv[1], *ld = argv[2], *ap = argv[3];
    int L = atoi(argv[4]);
    /* 位置参数只数到第一个 --flag 为止: 可选位置参数(K/INJ/XCAP/PREV)省略时 flag 会顶到
     * argv[5..], 不设界会把 "--ntok" 当 K 吃掉。 */
    int nfx = argc;
    for (int ai = 5; ai < argc; ai++) if (!strncmp(argv[ai], "--", 2)) { nfx = ai; break; }
    /* K/NTOK 必传(2026-08-31 魔数扫除): 旧静默默认 K=1024/NTOK=1716 与脚本恒传值
     * (冠军 K=64 / --ntok 8192)不一致, 手跑漏传会静默换口径 —— SCREEN_DIV 同款事故形态。 */
    if (nfx <= 5) die("K(z 秩)必传: 静默默认 1024 已删(冠军=64)");
    int K = atoi(argv[5]);
    int INJ = nfx > 6 ? atoi(argv[6]) : 1;
    const char *XCAP = (nfx > 7 && argv[7][0] && strcmp(argv[7], "-")) ? argv[7] : NULL;
    const char *PREV = (nfx > 8 && argv[8][0] && strcmp(argv[8], "-")) ? argv[8] : NULL;

    /* 二期支路与解算参数。原 DS4_ZL_* env(2026-08-31 禁 env 铁律清退), 现为跟在位置参数
     * 后的 --flag; 缺省值 = 原 env 不设时的行为, 逐个未动。ADDON 的"没 XANCHOR 就不生效"
     * 是 .py 权威口径(它把 ADDON 的读取整块写在 if XAP: 里面), 不"修正", 只多打一行提示。 */
    const char *GGP = NULL, *XAP = NULL, *GATEA = NULL;
    int ADDON = 0, CACHE_ONLY = 0, NGATE = 2048;
    int NTOK = 0, FTA = 1, GE_ON = 1, ERF_EN = 1, ERF_RANK = 8, GE_DEMEAN = 0;
    float SWLIM = 10.0f;
    double GATE = 0.0, GELAM = 1e-3, ERF_BAR = 0.01;
    for (int ai = nfx; ai < argc; ai++) {
        const char *a = argv[ai];
        const char *v = (ai + 1 < argc) ? argv[ai + 1] : NULL;
        if      (!strcmp(a, "--addon"))      ADDON = 1;
        else if (!strcmp(a, "--cache-only")) CACHE_ONLY = 1;
        else if (!strcmp(a, "--ge-demean"))  GE_DEMEAN = 1;   /* GE 针: 增益部署加权均值归一 */
        else if (!v) die("flag %s 缺值", a);
        else if (!strcmp(a, "--gguf"))    { GGP = *v ? v : NULL; ai++; }
        else if (!strcmp(a, "--xanchor")) { XAP = *v ? v : NULL; ai++; }
        else if (!strcmp(a, "--gate-anchor")) { GATEA = *v ? v : NULL; ai++; }
        else if (!strcmp(a, "--gate-rows"))   { NGATE = atoi(v); ai++; }
        else if (!strcmp(a, "--ntok"))    { NTOK = atoi(v); ai++; }
        else if (!strcmp(a, "--swlim"))   { SWLIM = (float)atof(v); ai++; }
        else if (!strcmp(a, "--fta"))     { FTA = atoi(v); ai++; }
        else if (!strcmp(a, "--ge"))      { GE_ON = atoi(v); ai++; }
        else if (!strcmp(a, "--gate"))    { GATE = atof(v); ai++; }
        else if (!strcmp(a, "--ge-lam"))  { GELAM = atof(v); ai++; }
        else if (!strcmp(a, "--erf"))     { ERF_EN = atoi(v); ai++; }
        else if (!strcmp(a, "--erf-bar")) { ERF_BAR = atof(v); ai++; }
        else if (!strcmp(a, "--erf-r"))   { ERF_RANK = atoi(v); ai++; }
        else die("不认识的 flag: %s", a);
    }
    if (NTOK <= 0) die("--ntok 必传(校准 token 行数): 静默默认 1716 已删(脚本恒传 8192)");
    if (ADDON && !XAP) {
        printf("  L%d 提示: 给了 --addon 但没给 --xanchor — .py 里 ADDON 的读取整块"
               "嵌在 if XAP: 内, 此时不生效, 本次照 .py 走非叠加路\n", L);
        ADDON = 0;
    }
    char path[1200];

    if (INJ == 2) {   /* 外挂模式断点续跑: zrec 已在则整层跳过(解算也省) */
        snprintf(path, sizeof path, "%s/zrec_L%02d.bin", ld, L);
        struct stat st;
        if (!stat(path, &st)) { printf("★L%d zrec 已存在(%lldB), 跳过\n", L, (long long)st.st_size); return 0; }
    }

    double t0 = now_s();
    float *X0fp = NULL, *rw = NULL; int *ridx = NULL; ameta_t am;
    anchor_layer(ap, L, NTOK, &X0fp, &ridx, &rw, &am);
    const int NACT = am.NACT;

    /* ★XCAP★ x 与被乘量都换成引擎真值(raw_ffn_in / raw_ffn_out), FP 侧仍走锚(教师)。
     * 教师路由 = FP 锚的 ridx/rw(2026-08-22 用户裁决: 换成量化路由等于换靶子)。
     * 没给 XCAP 目录 = 非 XCAP 口径: x 就是锚 fin(py: X0 保持 anchor_layer 的返回值),
     * 学生输出 Y_q 由本进程重算(VQ blob 或 GGUF 切片), 没有 YQE。 */
    float *X0 = XCAP ? capload(XCAP, "raw_ffn_in", L, NTOK, D) : X0fp;
    /* ★只换 x 模式(2026-08-27)★: XCAP 目录只给 raw_ffn_in 时 YQE=NULL —— 学生与教师
     * 都由本进程在【同一个量化链 x】上重算, 靶=纯量化误差(不掺引擎实现差)。用户口径:
     * "用量化链给反修用去对齐原始模型"。给了 raw_ffn_out 才走旧的引擎输出口径。 */
    float *YQE = NULL;
    if (XCAP) {
        char probe[1024];
        snprintf(probe, sizeof probe, "%s/raw_ffn_out_L%d", XCAP, L);
        FILE *pf = fopen(probe, "rb");
        if (pf) { fclose(pf); YQE = capload(XCAP, "raw_ffn_out", L, NTOK, D); }
        else printf("  L%d XCAP 只换 x 模式(无 raw_ffn_out): 学生/教师同 x 重算\n", L);
    }

    /* ★链态锚(--xanchor)★ 部署侧的 x_q 与路由_q 从第二个锚读; 教师侧仍用主锚。 */
    float *XQ0 = NULL, *rwq = NULL; int *ridxq = NULL;
    if (XAP) {
        ameta_t amq;
        anchor_layer(XAP, L, NTOK, &XQ0, &ridxq, &rwq, &amq);
        if (amq.NACT != am.NACT) die("链态锚 NACT=%d ≠ 主锚 %d — 槽宽不同, 口径不明拒跑", amq.NACT, am.NACT);
    }

    /* ★叠加式(--addon)★ 读本层 dql 里【已有】的记录: bf.GE 的每专家门 ge_old,
     * zl.RRR 的低秩 z_old。同名记录后出现的覆盖前面的(py 的循环就是这个语义)。 */
    float *ge_old = NULL;                       /* [NEXP] f32, NULL = 没有既有 GE */
    int zo_k0 = 0, zo_di = 0, zo_do = 0;        /* zo=(k0,di,do,z0,U0[do,k0],V0[di,k0]) */
    float *zo_z = NULL, *zo_U = NULL, *zo_V = NULL;
    if (ADDON) {
        snprintf(path, sizeof path, "%s/dql_L%02d.bin", ld, L);
        int dfd = open(path, O_RDONLY);
        if (dfd < 0) die("ADDON: dql 打不开 %s", path);
        struct stat dst2; fstat(dfd, &dst2);
        size_t dsz2 = (size_t)dst2.st_size;
        if (dsz2 < 12) die("ADDON: dql 太短 %s", path);
        const uint8_t *draw = (const uint8_t *)mmap(NULL, dsz2, PROT_READ, MAP_PRIVATE, dfd, 0);
        if (draw == MAP_FAILED) die("ADDON: dql mmap 失败");
        close(dfd);
        uint32_t nr2; memcpy(&nr2, draw + 8, 4);
        size_t off2 = 12;
        for (uint32_t i = 0; i < nr2; i++) {
            if (off2 + DS4_AMP_REC_HDR > dsz2) die("ADDON: dql 记录 %u 头越界", i);
            char nm2[17]; memcpy(nm2, draw + off2, 16); nm2[16] = 0;
            uint64_t psz2; memcpy(&psz2, draw + off2 + DS4_AMP_REC_OFF_PSZ, 8);
            int32_t vd; memcpy(&vd, draw + off2 + DS4_AMP_REC_OFF_VD, 4);
            if (off2 + DS4_AMP_REC_HDR + psz2 > dsz2) die("ADDON: dql 记录 %u 载荷越界", i);
            const uint8_t *pay2 = draw + off2 + DS4_AMP_REC_HDR;
            off2 += DS4_AMP_REC_HDR + (size_t)psz2;
            if (vd != 1) continue;
            if (strstr(nm2, "bf.GE") && psz2 >= 512) {
                if (!ge_old) ge_old = (float *)xmalloc(NEXP * 4);
                for (int e = 0; e < NEXP; e++) { uint16_t h; memcpy(&h, pay2 + e * 2, 2); ge_old[e] = f16_to_f32(h); }
            } else if (strstr(nm2, "zl.RRR") && psz2 >= 16) {
                uint32_t k0, di, do_; memcpy(&k0, pay2, 4); memcpy(&di, pay2 + 8, 4); memcpy(&do_, pay2 + 12, 4);
                size_t nh = (size_t)k0 + (size_t)k0 * do_ + (size_t)k0 * di;
                if (16 + nh * 2 > psz2) die("ADDON: zl.RRR 载荷不足(k=%u di=%u do=%u)", k0, di, do_);
                free(zo_z); free(zo_U); free(zo_V);
                zo_k0 = (int)k0; zo_di = (int)di; zo_do = (int)do_;
                zo_z = (float *)xmalloc((size_t)k0 * 4);
                zo_U = (float *)xmalloc((size_t)do_ * k0 * 4);
                zo_V = (float *)xmalloc((size_t)di * k0 * 4);
                const uint8_t *h = pay2 + 16;
                for (size_t j = 0; j < (size_t)k0; j++) { uint16_t v; memcpy(&v, h + j * 2, 2); zo_z[j] = f16_to_f32(v); }
                for (size_t j = 0; j < (size_t)do_ * k0; j++) { uint16_t v; memcpy(&v, h + (k0 + j) * 2, 2); zo_U[j] = f16_to_f32(v); }
                for (size_t j = 0; j < (size_t)di * k0; j++) { uint16_t v; memcpy(&v, h + (k0 + (size_t)do_ * k0 + j) * 2, 2); zo_V[j] = f16_to_f32(v); }
            }
        }
        munmap((void *)draw, dsz2);
        char zdesc[64];
        if (zo_z) snprintf(zdesc, sizeof zdesc, "k%d/din%d", zo_k0, zo_di); else snprintf(zdesc, sizeof zdesc, "无");
        printf("  L%d ADDON: 既有记录 GE=%s z=%s\n", L, ge_old ? "有" : "无", zdesc);
    }

