/* ds4quant_elm.inc.c — ELM 闭式乘性放大器解算(2026-08-28)
 *
 * 为什么要它: 今天 champ86 判决尺全线净负, 根因不是"z 不行", 是跑的解算器正好是记录里
 * 判过近零的那一档。fable5:5316 的三级实测(同一层 L35, 同一判据 held 行为挽回):
 *     加性闭式          0.0%
 *     乘性线性闭式     +0.6%
 *   ★乘性非线性 ELM  +12.02%★   ← 达被禁 SGD 版(13.4%)的 90%
 * 原话: "深层矿钥匙=非线性隐变量 z(线性 z 任何形态闭式全近零)"。
 * 现役 z_solve_dual 是【加性+线性】⇒ 实测 ZLGATE Δ 只有 0.02%, 与判例完全吻合。
 *
 * 模型(严格照 amp_probe3.py 定版, 八要件缺一不可):
 *     ŷ = y_q ⊙ (1 + g),  g = tanh(x·V₀/s) · U
 *     判据 = 1 − ‖y_fp−ŷ‖² / ‖y_fp−y_q‖²   (held 行为挽回)
 *   ①乘性(加性=0.0%) ②tanh(线性≈近零) ③s=sqrt(mean(Xa²)) 解析定标
 *   ④V₀ 双法 held 择优: PCA(fit 段主方向) vs 固定 seed 随机 —— 不人工选
 *   ⑤目标在行为空间比值域 R=dH·y_q/(y_q²+ε²), 不是直接拟合残差
 *   ⑥列权 colw=sqrt(var R)·sqrt(mean y_q²) (感知加权)
 *   ⑦dither 增广 Xa=[X, X+0.04·rms·randn] (正则)
 *   ⑧λ×k 双网格 held 每层自选 —— λ 固定过(全负)、k 固定过(被纠), 两条都在案
 *
 * 输入的 x 必须是【部署口径】(量化链 Fin), 不是锚的 FP 链态 x。fable5:6458 实锤: 口径错位
 * 时行 cos L40 只有 0.797(p5 到 0.63), z 是 x 的低秩函数 ⇒ 层内自评全绿、上链反噬。
 *
 * 复用同-TU 原语, 不另造轮子: cholesky/chol_solve_multi(zsolve)/zpar_for/dq_matmul(GPU)。
 */

#define ELM_KMAX 512
static const float ELM_LAM[5] = {100.0f, 30.0f, 10.0f, 3.0f, 1.0f};
static const int   ELM_KGRID[7] = {16, 32, 64, 128, 256, 384, 512};

typedef struct {
    int   from_pca;          /* 1=PCA 方向胜 0=随机特征胜(held 择优的结果) */
    float lam; int k;        /* held 选出的 λ 与 k */
    double held;             /* ★行有界行为挽回★(小数, 0.12 = 12%): 行权 1/(rowE+中位),
                              * 与 zlayer 四刀(2026-08-31)同式 —— 无界打分被重行主导,
                              * 深层薄肉被掩蔽是实测过的测量 bug, 探针必须同口径 */
    double held_uw;          /* 同配置的无界(旧口径)读数, 只为与 08-28 历史针对表 */
    double held_lin;         /* 同口径对照: 乘性【线性】(不过 tanh), 判例 +0.6% */
    float  s;                /* tanh 定标 */
    float *V0;               /* [D][k] 胜出方向(已裁到 k 列) */
    float *U;                /* [k][D] 已除 colw, 可直接用 */
} elm_res;

/* --elm-probe "2,20,40": 只在点名层跑针(CLI 参数, 不新增 env —— 铁律 2026-08-22) */
static const char *g_elm_probe = NULL;
static int elm_probe_hit(int L){
    if(!g_elm_probe) return 0;
    char b[256]; snprintf(b,sizeof b,"%s",g_elm_probe);
    for(char *t=strtok(b,","); t; t=strtok(NULL,",")) if(atoi(t)==L) return 1;
    return 0;
}
static void elm_free(elm_res *r){ if(!r) return; free(r->V0); free(r->U); r->V0=NULL; r->U=NULL; }

/* ── 并行核: 投影后的逐元素定标+激活 z = tanh(a/s)(投影本身交给生产 dq_matmul) ── */
typedef struct { float *Z; int K; float inv_s; int do_tanh; } elmt_ctx;
static void elmt_worker(void *vc, int r0, int r1){
    elmt_ctx *c=(elmt_ctx*)vc;
    for(int r=r0;r<r1;r++){
        float *z = c->Z + (size_t)r*c->K;
        if(c->do_tanh) for(int j=0;j<c->K;j++) z[j] = tanhf(z[j]*c->inv_s);
        else           for(int j=0;j<c->K;j++) z[j] = z[j]*c->inv_s;
    }
}
/* ── 并行核: Za = tanh(Xa·V0/s) ─────────────────────────────────────────── */
typedef struct { const float *Xa, *V0; float *Za; int D, K; float inv_s; int do_tanh; } elmz_ctx;
static void elmz_worker(void *vc, int r0, int r1){
    elmz_ctx *c = (elmz_ctx*)vc;
    for(int r=r0;r<r1;r++){
        const float *x = c->Xa + (size_t)r*c->D;
        float *z = c->Za + (size_t)r*c->K;
        for(int j=0;j<c->K;j++){
            const float *v = c->V0 + (size_t)j;     /* V0 是 [D][K] 行主序, 列 j 步长 K */
            double a = 0.0;
            for(int d=0;d<c->D;d++) a += (double)x[d]*v[(size_t)d*c->K];
            a *= c->inv_s;
            z[j] = c->do_tanh ? tanhf((float)a) : (float)a;
        }
    }
}

/* ── 并行核: 对称积 C = AᵀA (A 为 [n][m] 行主序, C 为 [m][m]), 按列块切 ──── */
typedef struct { const float *A; double *C; int n, m; } elmg_ctx;
static void elmg_worker(void *vc, int j0, int j1){
    elmg_ctx *c = (elmg_ctx*)vc;
    for(int j=j0;j<j1;j++)
        for(int k2=j;k2<c->m;k2++){
            const float *aj = c->A + j, *ak = c->A + k2;
            double s2 = 0.0;
            for(int i=0;i<c->n;i++) s2 += (double)aj[(size_t)i*c->m]*ak[(size_t)i*c->m];
            c->C[(size_t)j*c->m+k2] = s2;
        }
}
/* ── 并行核: B = AᵀR (A 为 [n][m], R 为 [n][D], B 为 [m][D]) ─────────────── */
typedef struct { const float *A; const double *R; double *B; int n, m, D; } elmb_ctx;
static void elmb_worker(void *vc, int j0, int j1){
    elmb_ctx *c = (elmb_ctx*)vc;
    for(int j=j0;j<j1;j++){
        double *b = c->B + (size_t)j*c->D;
        for(int d=0;d<c->D;d++) b[d] = 0.0;
        for(int i=0;i<c->n;i++){
            const double a = c->A[(size_t)i*c->m + j];
            if(a == 0.0) continue;
            const double *r = c->R + (size_t)i*c->D;
            for(int d=0;d<c->D;d++) b[d] += a*r[d];
        }
    }
}

/* ── 并行核: k 前缀秩-1 累加 gac[i] += Ze[i][c]·U[c] (按 held 行切) ────── */
typedef struct { const float *Ze; const double *u; double *gac; int K, D, c; } elma_ctx;
static void elma_worker(void *vc, int i0, int i1){
    elma_ctx *a = (elma_ctx*)vc;
    for(int i=i0;i<i1;i++){
        const double z = a->Ze[(size_t)i*a->K + a->c];
        if(z == 0.0) continue;
        double *ga = a->gac + (size_t)i*a->D;
        for(int d=0;d<a->D;d++) ga[d] += z*a->u[d];
    }
}

/* ── PCA: XaᵀXa 的 top-K 特征子空间(块子空间迭代, 复用 zsub_* 那套的思路) ──
 * 逐个 deflation 求 512 个特征对是 O(k²n), 太慢; 块幂迭代一次拿整个子空间。
 * PCA 方向只是 tanh 的特征基(ELM 随机特征同族), 近似子空间足够 —— 何况最终还要
 * 和"固定 seed 随机方向"在 held 上比一场, approximation 好不好由 held 说了算。 */
/* ★块幂迭代走生产 GEMM(铁律: 只有 GPU 版本)★ 12 轮 × D²K ≈ 1e11, 自己写循环即使 20 线程
 * 也要几十秒。M 是【全对称阵】(dq_matmul 出的完整 D×D, 不是上三角), 所以
 * T[D,K] = M[D,D]·V[D,K] 可以直接 dq_matmul(A=M[D,D], B=VT[K,D], out=T[D,K]) = M·VTᵀ。
 * V 在 mgs 里是 [D][K] 布局, 故每轮转一次 VT[K][D](D·K=2M 元素, 忽略不计)。 */
static void elm_pca_dirs(const float *M, int D, int K, float *V0, uint64_t *seed){
    float *V = malloc((size_t)D*K*4), *T = malloc((size_t)D*K*4), *VT = malloc((size_t)K*D*4);
    for(size_t i=0;i<(size_t)D*K;i++) V[i] = lcg_unit(seed);
    mgs(V, (uint32_t)D, (uint32_t)K, seed);
    for(int it=0; it<12; it++){
        for(int d=0;d<D;d++) for(int c=0;c<K;c++) VT[(size_t)c*D+d] = V[(size_t)d*K+c];
        dq_matmul(M, VT, T, D, D, K);                /* T[D,K] = M[D,D]·VT[K,D]ᵀ */
        memcpy(V, T, (size_t)D*K*4);
        mgs(V, (uint32_t)D, (uint32_t)K, seed);
    }
    memcpy(V0, V, (size_t)D*K*4);
    free(V); free(T); free(VT);
}

/* ── 主解算 ─────────────────────────────────────────────────────────────────
 * X  [S][D]  有效 x(量化链 Fin)
 * YQ [S][D]  量化态 routed 输出
 * DH [S][D]  y_fp − y_q (FP routed − 量化 routed)
 * vs         fit 行数(前 vs 行拟合, 其余 held 评判)
 * 返回 0=成功。失败时 out 不可用。 */
static int elm_solve(const float *X, const float *YQ, const float *DH,
                     int S, int D, int vs, elm_res *out)
{
    const int ev0 = vs, nev = S - vs, ntr = vs;
    if(nev < 64 || ntr < 64 || D <= 0) return -1;
    const int K = ELM_KMAX < D ? ELM_KMAX : D;
    const int na = 2*ntr;                            /* dither 增广后的行数 */
    memset(out, 0, sizeof(*out));

    /* ①② 行为空间比值域目标 R 与列权 colw(只用 fit 行统计) */
    double *R = malloc((size_t)S*D*sizeof(double));
    float  *eps = malloc((size_t)D*4), *colw = malloc((size_t)D*4);
    for(int d=0; d<D; d++){
        double m2 = 0.0;
        for(int i=0;i<ntr;i++){ double v = YQ[(size_t)i*D+d]; m2 += v*v; }
        m2 /= ntr;
        eps[d] = (float)(sqrt(m2)*1e-2 + 1e-12);
        (void)0;
    }
    for(size_t i=0;i<(size_t)S*D;i++){
        const int d = (int)(i % (size_t)D);
        const double yq = YQ[i], e = eps[d];
        R[i] = DH[i]*yq/(yq*yq + e*e);
    }
    for(int d=0; d<D; d++){
        double s1=0, s2=0, q2=0;
        for(int i=0;i<ntr;i++){ double r = R[(size_t)i*D+d]; s1+=r; s2+=r*r;
                                double y = YQ[(size_t)i*D+d]; q2+=y*y; }
        const double var = s2/ntr - (s1/ntr)*(s1/ntr);
        colw[d] = (float)(sqrt(var>0?var:0.0 + 1e-12) * sqrt(q2/ntr + 1e-12));
    }

    /* ③⑦ dither 增广 + tanh 解析定标 */
    float *Xa = malloc((size_t)na*D*4);
    double *Ra = malloc((size_t)na*D*sizeof(double));
    uint64_t sd = 0x9E3779B97F4A7C15ULL;
    for(int i=0;i<ntr;i++){
        const float *x = X + (size_t)i*D;
        double rr = 0.0; for(int d=0;d<D;d++) rr += (double)x[d]*x[d];
        const float rms = (float)sqrt(rr/D);
        memcpy(Xa + (size_t)i*D, x, (size_t)D*4);
        float *xd = Xa + (size_t)(ntr+i)*D;
        for(int d=0;d<D;d++){                        /* 0.04·rms 高斯噪声(Box-Muller) */
            const double u1 = lcg_unit(&sd)*0.5+0.5, u2 = lcg_unit(&sd)*0.5+0.5;
            const double g = sqrt(-2.0*log(u1>1e-12?u1:1e-12))*cos(6.283185307179586*u2);
            xd[d] = x[d] + (float)(0.04*rms*g);
        }
        const double *r = R + (size_t)i*D;
        double *ra0 = Ra + (size_t)i*D, *ra1 = Ra + (size_t)(ntr+i)*D;
        for(int d=0;d<D;d++){ ra0[d] = r[d]*colw[d]; ra1[d] = ra0[d]; }
    }
    double sc2 = 0.0;
    for(size_t i=0;i<(size_t)na*D;i++) sc2 += (double)Xa[i]*Xa[i];
    const float scale = (float)sqrt(sc2/((double)na*D));
    out->s = scale;

    /* 基线误差 e0 = ‖y_fp − y_q‖² on held(y_fp = y_q + dH ⇒ 就是 ‖dH‖²)。
     * 同时算行有界权 rowW=1/(rowE+中位): 无界求和被少数重行主导(zlayer 四刀实锤的
     * 测量 bug), 择优与主读数一律行有界, 无界只留对表。 */
    double e0 = 0.0, e0w = 0.0;
    double *rowE = malloc((size_t)nev*sizeof(double));
    double *rowW = malloc((size_t)nev*sizeof(double));
    for(int i=0;i<nev;i++){ const float *dh = DH + (size_t)(ev0+i)*D;
        double s2r = 0.0; for(int d=0;d<D;d++) s2r += (double)dh[d]*dh[d];
        rowE[i] = s2r; e0 += s2r; }
    if(e0 <= 0.0){ free(R);free(eps);free(colw);free(Xa);free(Ra);free(rowE);free(rowW); return -1; }
    {   double *srt = malloc((size_t)nev*sizeof(double));
        memcpy(srt, rowE, (size_t)nev*sizeof(double));
        for(int i=1;i<nev;i++){ double v=srt[i]; int j=i-1;   /* 插入排序: nev≈1.5k, 一次性 */
            while(j>=0 && srt[j]>v){ srt[j+1]=srt[j]; j--; } srt[j+1]=v; }
        const double med = srt[nev/2] + 1e-30; free(srt);
        for(int i=0;i<nev;i++){ rowW[i] = 1.0/(rowE[i]+med); e0w += rowW[i]*rowE[i]; }
    }

    /* ④ 两套 V₀: PCA(Xa 去均值的二阶矩 top-K) 与 固定 seed 随机 */
    float *Vp = malloc((size_t)D*K*4), *Vr = malloc((size_t)D*K*4);
    {   /* ★协方差走生产 GEMM(铁律 2026-08-28 只有 GPU 版本)★
         * 原来是 elmg_worker 的 D²/2×na ≈ 1.03e11 次 double 标量乘加(20 线程也要 40 秒),
         * 换 dq_matmul(带 -DDS4QUANT_CUDA 走 cuBLAS, 否则走 BLAS sgemm) 一句解决。
         * dq_matmul(A[S,K],B[M,K],out[S,M]) = A·Bᵀ ⇒ 要 XcT[D][na], 转置一次(200MB)。 */
        float *mu = calloc((size_t)D, 4);
        for(int i=0;i<na;i++){ const float *x=Xa+(size_t)i*D; for(int d=0;d<D;d++) mu[d]+=x[d]; }
        for(int d=0;d<D;d++) mu[d] /= (float)na;
        float *XcT = malloc((size_t)D*na*4);
        for(int i=0;i<na;i++){ const float *x=Xa+(size_t)i*D;
            for(int d=0;d<D;d++) XcT[(size_t)d*na+i] = x[d]-mu[d]; }
        free(mu);
        float *M = malloc((size_t)D*D*4);
        dq_matmul(XcT, XcT, M, D, na, D);      /* M[D,D] = XcT·XcTᵀ = XcᵀXc */
        free(XcT);
        uint64_t s3 = 0x5A5A1EEDULL;
        elm_pca_dirs(M, D, K, Vp, &s3);
        free(M);
    }
    { uint64_t s4 = 7ULL; const float inv = 1.0f/sqrtf((float)D);
      for(size_t i=0;i<(size_t)D*K;i++) Vr[i] = lcg_unit(&s4)*inv; }

    /* ⑤⑥⑧ 对两套 V₀ 各扫 λ×k, held 择优 */
    float *Za = malloc((size_t)na*K*4), *Ze = malloc((size_t)nev*K*4);
    /* GEMM 要的转置视图: dq_matmul(A[S,K],B[M,K]) 算 A·Bᵀ, 所以右操作数得是 [M][K] 布局 */
    float *V0T = malloc((size_t)K*D*4);      /* V₀[D][K] → [K][D] */
    float *ZaT = malloc((size_t)K*na*4);     /* Za[na][K] → [K][na] */
    float *RaT = malloc((size_t)D*na*4);     /* Ra[na][D] → [D][na], 且 double→float */
    for(int i=0;i<na;i++) for(int d=0;d<D;d++) RaT[(size_t)d*na+i] = (float)Ra[(size_t)i*D+d];
    double *ZtZ = malloc((size_t)K*K*sizeof(double));
    double *ZtR = malloc((size_t)K*D*sizeof(double));
    double *G   = malloc((size_t)K*K*sizeof(double));
    double *Ub  = malloc((size_t)K*D*sizeof(double));
    float  *gacf = malloc((size_t)nev*D*4);              /* 网格点增益(GEMM 输出) */
    float  *UbT  = malloc((size_t)D*K*4);                /* Ub[K][D] 的转置视图, GEMM 右操作数 */
    float  *ZeK  = malloc((size_t)nev*K*4);              /* Ze 的前 k 列紧凑副本(行距要连续) */
    out->held = -1e300;

    for(int variant=0; variant<2; variant++){
        const float *V0 = variant ? Vr : Vp;
        for(int lin=0; lin<(variant?1:2); lin++){        /* lin=1 只在 PCA 支跑一次(线性对照) */
            const int do_tanh = !lin;
            /* ★投影走生产 GEMM★(铁律): Za=Xa·V₀ 是 na×D×K≈2.6e10, 自写循环是 CPU 大头。
             * dq_matmul 要 B 为 [M][K] 布局 ⇒ V₀T[K][D]; 之后 tanh 逐元素(便宜, 留 CPU)。 */
            for(int d=0;d<D;d++) for(int c=0;c<K;c++) V0T[(size_t)c*D+d]=V0[(size_t)d*K+c];
            dq_matmul(Xa, V0T, Za, na, D, K);
            dq_matmul(X+(size_t)ev0*D, V0T, Ze, nev, D, K);
            { elmt_ctx tc={Za,K,1.0f/scale,do_tanh}; zpar_for(na, 20, elmt_worker, &tc); }
            { elmt_ctx tc={Ze,K,1.0f/scale,do_tanh}; zpar_for(nev,20, elmt_worker, &tc); }
            /* ZtZ = ZaᵀZa, ZtR = ZaᵀRa —— 同样走 GEMM(要 Zaᵀ[K][na] 与 Raᵀ[D][na]) */
            for(int i=0;i<na;i++) for(int c=0;c<K;c++) ZaT[(size_t)c*na+i]=Za[(size_t)i*K+c];
            { float *tz=malloc((size_t)K*K*4);
              dq_matmul(ZaT, ZaT, tz, K, na, K);
              for(size_t i=0;i<(size_t)K*K;i++) ZtZ[i]=tz[i]; free(tz); }
            { float *tr=malloc((size_t)K*D*4);
              dq_matmul(ZaT, RaT, tr, K, na, D);
              for(size_t i=0;i<(size_t)K*D;i++) ZtR[i]=tr[i]; free(tr); }
            double trz = 0.0; for(int a=0;a<K;a++) trz += ZtZ[(size_t)a*K+a];

            for(int li=0; li<5; li++){
                const double lam = ELM_LAM[li];
                memcpy(G, ZtZ, (size_t)K*K*sizeof(double));
                for(int a=0;a<K;a++) G[(size_t)a*K+a] += lam*trz/K + 1e-10;
                if(cholesky(G,(uint32_t)K) != 0) continue;
                /* U = G⁻¹·ZᵀR。★走 BLAS 三角求解, 不用自写回代★(铁律: 只有 GPU 版本)
                 * 原来是 chol_solve_multi 按 16 列分块自写前代/回代: K²×D×5λ×3支 ≈ 1.6e10 次
                 * 标量运算 + 每块一次 malloc/两次拷贝, 是反修剩下的最后一个 CPU 大头。
                 * cholesky() 把 L 写在【行主序下三角】(A[i*d+j], i≥j), 正是 CblasLower 的口径;
                 * 一次 dtrsm 解 L·Y=B, 再一次(Trans)解 Lᵀ·X=Y, 全部 D 列一起做, 零分块零拷贝。 */
                memcpy(Ub, ZtR, (size_t)K*D*sizeof(double));
                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, K, D, 1.0, G, K, Ub, D);
                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans,
                            CblasNonUnit, K, D, 1.0, G, K, Ub, D);
                /* k 是前缀: 按列秩-1 累加, 在网格点上评 held(免得每个 k 重算整个矩阵乘) */
                /* ★k 网格改按点 GEMM(铁律: 只有 GPU 版本)★ 原来是"按列秩-1 累加, 到网格点取快照",
                 * 省 flops 但全是 CPU 标量(K×nev×D×5λ×3支 ≈ 1.6e10, 实测占反修剩余时间一半)。
                 * 改成每个网格点直接 g_k = Ze[:, :k]·U[:k,:] 一次 GEMM: flops 多 2.7 倍, 但走
                 * 生产 dq_matmul(cuBLAS/BLAS)后净快得多。★数值等价★: 同样是前 k 列的和。 */
                for(int gi=0; gi<7; gi++){
                    const int kk = ELM_KGRID[gi];
                    if(kk > K) break;
                    /* dq_matmul(A[S,K],B[M,K]) = A·Bᵀ ⇒ 要 UbT[D][kk](Ub 是 [K][D]) */
                    for(int d=0;d<D;d++) for(int c=0;c<kk;c++) UbT[(size_t)d*kk+c]=(float)Ub[(size_t)c*D+d];
                    for(int i=0;i<nev;i++) memcpy(ZeK+(size_t)i*kk, Ze+(size_t)i*K, (size_t)kk*4);
                    dq_matmul(ZeK, UbT, gacf, nev, kk, D);  /* g[nev,D] = ZeK[nev,kk]·UbT[D,kk]ᵀ */
                    double err = 0.0, errw = 0.0;
                    for(int i=0;i<nev;i++){
                        const float *yq = YQ + (size_t)(ev0+i)*D, *dh = DH + (size_t)(ev0+i)*D;
                        const float *ga = gacf + (size_t)i*D;
                        double rerr = 0.0;
                        for(int d=0;d<D;d++){
                            const double g2 = (double)ga[d]/colw[d];
                            const double e2 = (double)dh[d] - (double)yq[d]*g2;  /* y_fp−ŷ */
                            rerr += e2*e2;
                        }
                        err += rerr; errw += rowW[i]*rerr;
                    }
                    const double hv = 1.0 - errw/e0w;        /* 行有界 = 择优判据 */
                    const double hv_uw = 1.0 - err/e0;       /* 无界只对表不择优 */
                    if(lin){ if(hv > out->held_lin) out->held_lin = hv; continue; }
                    if(hv > out->held){
                        out->held = hv; out->held_uw = hv_uw; out->lam = (float)lam; out->k = kk;
                        out->from_pca = !variant;
                        free(out->V0); free(out->U);
                        out->V0 = malloc((size_t)D*kk*4);
                        out->U  = malloc((size_t)kk*D*4);
                        for(int d=0;d<D;d++) for(int c2=0;c2<kk;c2++)
                            out->V0[(size_t)d*kk+c2] = V0[(size_t)d*K+c2];
                        for(int c2=0;c2<kk;c2++) for(int d=0;d<D;d++)
                            out->U[(size_t)c2*D+d] = (float)(Ub[(size_t)c2*D+d]/colw[d]);
                    }
                }
            }
        }
    }
    free(R);free(eps);free(colw);free(Xa);free(Ra);free(Vp);free(Vr);
    free(Za);free(Ze);free(ZtZ);free(ZtR);free(G);free(Ub);free(gacf);free(UbT);free(ZeK);
    free(V0T);free(ZaT);free(RaT);free(rowE);free(rowW);
    return out->V0 ? 0 : -1;
}
