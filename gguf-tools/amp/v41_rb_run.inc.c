/* v41_rb_run.inc.c — 路由偏置侧车(路由反修)在 V4.1 上的拟合驱动(2026-09-20)。v41_amp_run.c 单 TU include, 拆出只为守 500 行;
 * 数值在 v41_route_solve.cu; 引擎只多一个加载点(rb_Lnn.bin → 该层路由选择分加 Δb: core_v41_amp.c / cuda_v41_3.inc.cu)。
 *
 * 【V4 原型】ds4quant_run_p4.inc.c 的路由偏置侧车(2026-07-28): 神谕探针实证反修字节的亏损全住在路由漂移; FIT = 学生路由回放时
 * 按层按专家累计"FP 锚选中集合 vs 学生集合"的选择分 margin 缺口(漏选 += thr−v, 多选 −= v−thr, thr = 学生第 K 名选择分),
 * 均值 + 计数(≥8 才武装) → Δb; APPLY = α·Δb 只加进选择分, 权重分不动(与引擎 exp_probs_b 同语义)。冠军 α=2.5 是拍的。
 * 【这一版】一趟前向、钩子每层回调(非序贯统计, 与 V4 固定档模式同): 拟合行累计缺口, val 行留存学生选择分 + FP 集合;
 * 趟末每层解 Δb, α 网格按 val 行"与 FP 同集"率择优, 只有严格涨的层落 rb_Lnn.bin(文件里已含 α)。
 * 【FP 路由从哪来】教师 --dump-moe 的 x_Lnn.bin(FP 链态输入), 工具里重算 router(x_fp): gate 权重原生 bf16, 与 FP 模型同一份 ⇒ 就是 FP 的真路由。
 * 【硬闸】工具重算的学生路由与钩子 sel 同集 ≥ 99% —— 重算口径不是引擎的, 后面全是假账。 */
#define RB_NA 4
static const float RB_ALPHAS[RB_NA] = {0.5f, 1.0f, 1.5f, 2.5f};
#define RB_MINCNT 8
#define RB_SEL_CHECK 0.99

typedef struct {
    ctx_t *c; int nl, E, K, D, n, nval, cap, bad;
    const char *dump; FILE **fx;                 /* [nl] x_Lnn.bin 句柄(懒开) */
    float **dWg, **dB;                           /* [nl] 设备路由权重(懒传) */
    double *acc; long *cnt, *nq, *nf;            /* [nl][E] margin 缺口 / 计数 / 学生与 FP 的选中计数(全行) */
    long *chk_same, *chk_rows, *fit_same, *fit_rows, *val_same, *val_rows;   /* [nl] */
    float *vsc; int *vself; int *vslot;          /* val 留存 [nl][nval][E] 学生选择分 / [nl][nval][K] FP 集合; vslot[全局行] = val 槽(-1 = 拟合行) */
    float *dXc, *dXf, *dw, *dwf, *dsc, *dscf; int *dsel, *dself;   /* 一块的设备暂存(cap 行) */
    float *hxf, *hsc; int *hsel, *hself;         /* 主机暂存 */
    float scale; int scale_set;
} rb_ctx;
static rb_ctx g_rb;

static int rb_has(const int *s, int K, int e) { for (int k = 0; k < K; k++) if (s[k] == e) return 1; return 0; }
static int rb_overlap(const int *a, const int *b, int K) { int o = 0; for (int k = 0; k < K; k++) if (rb_has(b, K, a[k])) o++; return o; }

static int rb_grow(rb_ctx *r, int n) {
    if (n <= r->cap) return 0;
    const int cap = n + 64; const size_t nD = (size_t)cap * r->D, nK = (size_t)cap * r->K, nE = (size_t)cap * r->E;
    cudaFree(r->dXc); cudaFree(r->dXf); cudaFree(r->dw); cudaFree(r->dwf); cudaFree(r->dsc); cudaFree(r->dscf); cudaFree(r->dsel); cudaFree(r->dself);
    free(r->hxf); free(r->hsc); free(r->hsel); free(r->hself);
    r->hxf = malloc(nD * 4); r->hsc = malloc(nE * 4); r->hsel = malloc(nK * 4); r->hself = malloc(nK * 4);
    if (!r->hxf || !r->hsc || !r->hsel || !r->hself || cudaMalloc((void **)&r->dXc, nD * 4) || cudaMalloc((void **)&r->dXf, nD * 4) ||
        cudaMalloc((void **)&r->dw, nK * 4) || cudaMalloc((void **)&r->dwf, nK * 4) || cudaMalloc((void **)&r->dsc, nE * 4) || cudaMalloc((void **)&r->dscf, nE * 4) ||
        cudaMalloc((void **)&r->dsel, nK * 4) || cudaMalloc((void **)&r->dself, nK * 4)) { fprintf(stderr, "★路由侧车暂存分配失败★\n"); return -1; }
    r->cap = cap;
    return 0;
}

/* 钩子: 每层每块回调一次(pos0 = 块起始行)。学生路由在工具里重算(要全部 E 个选择分), FP 路由由 x_fp 重算。 */
static int rb_hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel, const float *rw,
                   const float *alpha, const float *ye, const float *ysh) {
    rb_ctx *r = &g_rb; ctx_t *c = ud; (void)y; (void)clamp; (void)alpha; (void)ye; (void)ysh;
    if (r->bad) return -1;
    if (il < 0 || il >= r->nl) return 0;
    if (r->K < 0) r->K = n_used;   /* K 只有钩子给, 第一次回调定下, 之后每次核 */
    if (D != r->D || n_used != r->K || n_used > MAXU || pos0 < 0 || pos0 + n > r->n) { fprintf(stderr, "★rb 钩子 L%02d 形状不符(D %d K %d pos0 %d n %d)★\n", il, D, n_used, pos0, n); r->bad = 1; return -1; }
    if (!r->scale_set) { double s = 0; for (int k = 0; k < n_used; k++) s += rw[k]; r->scale = (float)s; r->scale_set = 1; }   /* Σ_k w = route_scale(归一化后) */
    if (rb_grow(r, n)) { r->bad = 1; return -1; }
    const int E = r->E, K = r->K;
    if (!r->dWg[il]) {   /* 路由权重从 GGUF 取(与引擎同一份字节) */
        char nm[96]; snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", il);
        const ds4_gguf_tensor *tg = ds4_gguf_find(&c->gq, nm);
        snprintf(nm, sizeof nm, "blk.%d.exp_probs_b.bias", il);
        const ds4_gguf_tensor *tb = ds4_gguf_find(&c->gq, nm);
        uint64_t gb = 0, bb = 0;
        const uint8_t *pg = tg ? ds4_gguf_tensor_data(&c->gq, tg, &gb) : NULL, *pb = tb ? ds4_gguf_tensor_data(&c->gq, tb, &bb) : NULL;
        if (!pg || !pb || tg->type != DS4_GGT_BF16 || tb->type != DS4_GGT_F32 || gb != (uint64_t)E * D * 2 || bb != (uint64_t)E * 4 ||
            v41_route_upload_gate_gpu((const uint16_t *)pg, (const float *)pb, E, D, &r->dWg[il], &r->dB[il])) {
            fprintf(stderr, "★L%02d 路由张量缺/型不对/上传失败★\n", il); r->bad = 1; return -1;
        }
    }
    if (!r->fx[il]) {
        char p[4300]; snprintf(p, sizeof p, "%s/x_L%02d.bin", r->dump, il);
        r->fx[il] = fopen(p, "rb");
        if (!r->fx[il]) { fprintf(stderr, "★教师 dump 缺 %s★\n", p); r->bad = 1; return -1; }
        fseek(r->fx[il], 0, SEEK_END);
        if (ftell(r->fx[il]) != (long)((size_t)r->n * D * 4)) { fprintf(stderr, "★%s 大小与取料行数不符★\n", p); r->bad = 1; return -1; }
    }
    if (fseek(r->fx[il], (long)((size_t)pos0 * D * 4), SEEK_SET) || fread(r->hxf, 4, (size_t)n * D, r->fx[il]) != (size_t)n * D) { r->bad = 1; return -1; }
    if (cudaMemcpy(r->dXc, x, (size_t)n * D * 4, cudaMemcpyHostToDevice) || cudaMemcpy(r->dXf, r->hxf, (size_t)n * D * 4, cudaMemcpyHostToDevice) ||
        v41_route_recompute_gpu(r->dXc, r->dWg[il], r->dB[il], NULL, n, D, E, K, r->scale, r->dsel, r->dw, r->dsc) ||
        v41_route_recompute_gpu(r->dXf, r->dWg[il], r->dB[il], NULL, n, D, E, K, r->scale, r->dself, r->dwf, r->dscf) ||
        cudaMemcpy(r->hsel, r->dsel, (size_t)n * K * 4, cudaMemcpyDeviceToHost) || cudaMemcpy(r->hself, r->dself, (size_t)n * K * 4, cudaMemcpyDeviceToHost) ||
        cudaMemcpy(r->hsc, r->dsc, (size_t)n * E * 4, cudaMemcpyDeviceToHost)) { fprintf(stderr, "★L%02d 路由重算失败★\n", il); r->bad = 1; return -1; }
    double *acc = r->acc + (size_t)il * E; long *cnt = r->cnt + (size_t)il * E, *nq = r->nq + (size_t)il * E, *nf = r->nf + (size_t)il * E;
    for (int t = 0; t < n; t++) {
        const int g = pos0 + t; const int *a = r->hsel + (size_t)t * K, *b = r->hself + (size_t)t * K; const float *s = r->hsc + (size_t)t * E;
        r->chk_rows[il]++; if (rb_overlap(a, sel + (size_t)t * K, K) == K) r->chk_same[il]++;
        const int ov = rb_overlap(a, b, K);
        for (int k = 0; k < K; k++) { nq[a[k]]++; nf[b[k]]++; }
        const int slot = r->vslot[g];
        if (slot < 0) {
            r->fit_rows[il]++; if (ov == K) r->fit_same[il]++;
            float thr = s[a[0]]; for (int k = 1; k < K; k++) if (s[a[k]] < thr) thr = s[a[k]];
            for (int k = 0; k < K; k++) { const int e = b[k]; if (!rb_has(a, K, e)) { acc[e] += thr - s[e]; cnt[e]++; } }
            for (int k = 0; k < K; k++) { const int e = a[k]; if (!rb_has(b, K, e)) { acc[e] -= s[e] - thr; cnt[e]++; } }
        } else {
            r->val_rows[il]++; if (ov == K) r->val_same[il]++;
            memcpy(r->vsc + ((size_t)il * r->nval + slot) * E, s, (size_t)E * 4);
            memcpy(r->vself + ((size_t)il * r->nval + slot) * K, b, (size_t)K * 4);
        }
    }
    return 0;
}

/* val 行上挂 α·Δb 重选 top-K, 数与 FP 同集的行数 */
static long rb_val_same(const rb_ctx *r, int il, const float *db, float alpha) {
    const int E = r->E, K = r->K; long same = 0; int pick[16];
    for (int v = 0; v < r->nval; v++) {
        const float *s = r->vsc + ((size_t)il * r->nval + v) * E; const int *b = r->vself + ((size_t)il * r->nval + v) * K;
        for (int k = 0; k < K; k++) {
            float bv = -INFINITY; int be = -1;
            for (int e = 0; e < E; e++) { if (rb_has(pick, k, e)) continue; const float sv = s[e] + alpha * db[e]; if (sv > bv) { bv = sv; be = e; } }
            pick[k] = be;
        }
        if (rb_overlap(pick, b, K) == K) same++;
    }
    return same;
}

static int rb_solve_layer(rb_ctx *r, int il, FILE *mf, const char *out_dir, int *nw) {
    const int E = r->E, K = r->K;
    const double chk = r->chk_rows[il] ? (double)r->chk_same[il] / r->chk_rows[il] : 0.0;
    float *db = malloc((size_t)E * 4); int armed = 0;
    for (int e = 0; e < E; e++) { const long c0 = r->cnt[(size_t)il * E + e]; db[e] = c0 >= RB_MINCNT ? (float)(r->acc[(size_t)il * E + e] / c0) : 0.f; if (db[e] != 0.f) armed++; }
    long sysd = 0; for (int e = 0; e < E; e++) sysd += labs(r->nq[(size_t)il * E + e] - r->nf[(size_t)il * E + e]);
    const double base = r->val_rows[il] ? 100.0 * r->val_same[il] / r->val_rows[il] : 0.0, fitr = r->fit_rows[il] ? 100.0 * r->fit_same[il] / r->fit_rows[il] : 0.0;
    double rate[RB_NA]; int bi = -1;
    for (int ai = 0; ai < RB_NA; ai++) { rate[ai] = r->val_rows[il] ? 100.0 * rb_val_same(r, il, db, RB_ALPHAS[ai]) / r->val_rows[il] : 0.0; if (bi < 0 || rate[ai] > rate[bi]) bi = ai; }
    const int accept = chk >= RB_SEL_CHECK && armed > 0 && rate[bi] > base;
    printf("[L%02d] 自检同集 %.2f%% | 与 FP 同集 拟合 %.1f%% val %.1f%% | 系统性错位 %.2f%% 的 pick | 武装 %d/%d | α 0.5:%.1f 1:%.1f 1.5:%.1f 2.5:%.1f → %s\n",
           il, 100.0 * chk, fitr, base, 50.0 * sysd / ((double)r->chk_rows[il] * K), armed, E, rate[0], rate[1], rate[2], rate[3],
           accept ? "挂" : (chk < RB_SEL_CHECK ? "★自检不过★" : "不挂(val 不涨)"));
    fprintf(mf, "L%02d rb %s α=%g 自检 %.4f val同集 %.2f→%.2f 拟合 %.2f 武装 %d 系统 %.2f\n", il, accept ? "挂" : "skip", RB_ALPHAS[bi], chk, base, rate[bi], fitr, armed, 50.0 * sysd / ((double)r->chk_rows[il] * K));
    if (chk < RB_SEL_CHECK) { fprintf(stderr, "★L%02d 重算路由与钩子只有 %.2f%% 同集 —— 口径不是引擎的, 停车★\n", il, 100.0 * chk); free(db); return -1; }
    if (!accept) { free(db); return 0; }
    for (int e = 0; e < E; e++) db[e] *= RB_ALPHAS[bi];
    char p[4300], q[4300]; snprintf(p, sizeof p, "%s/rb_L%02d.bin.part", out_dir, il); snprintf(q, sizeof q, "%s/rb_L%02d.bin", out_dir, il);
    FILE *f = fopen(p, "wb");
    if (!f) { free(db); return -1; }
    const int32_t hd[2] = { E, 1 };
    const int okw = fwrite(hd, 4, 2, f) == 2 && fwrite(db, 4, (size_t)E, f) == (size_t)E;
    free(db);
    if (fclose(f) || !okw || rename(p, q)) { fprintf(stderr, "★rb_L%02d.bin 落盘失败★\n", il); remove(p); return -1; }
    (*nw)++;
    return 1;
}

/* 统计表分配(两种模式共用): 一趟模式一次统计全部层; 合并序贯每遍只统计第 k 层。表按 [nl][E] 开, 层与层互不干扰。 */
static int rb_seq_init(ctx_t *c, int ntok, int nl) {
    rb_ctx *r = &g_rb; memset(r, 0, sizeof *r);
    r->c = c; r->nl = nl; r->E = c->n_expert; r->D = c->D; r->n = ntok; r->dump = c->rb_dump; r->nval = ntok - c->nfit;
    r->fx = calloc((size_t)nl, sizeof(FILE *)); r->dWg = calloc((size_t)nl, sizeof(float *)); r->dB = calloc((size_t)nl, sizeof(float *));
    const size_t nlE = (size_t)nl * r->E;
    r->acc = calloc(nlE, sizeof(double)); r->cnt = calloc(nlE, sizeof(long)); r->nq = calloc(nlE, sizeof(long)); r->nf = calloc(nlE, sizeof(long));
    r->chk_same = calloc((size_t)nl, sizeof(long)); r->chk_rows = calloc((size_t)nl, sizeof(long)); r->fit_same = calloc((size_t)nl, sizeof(long));
    r->fit_rows = calloc((size_t)nl, sizeof(long)); r->val_same = calloc((size_t)nl, sizeof(long)); r->val_rows = calloc((size_t)nl, sizeof(long));
    r->vslot = malloc(sizeof(int) * (size_t)ntok);
    if (!r->fx || !r->dWg || !r->dB || !r->acc || !r->cnt || !r->nq || !r->nf || !r->chk_same || !r->chk_rows || !r->fit_same || !r->fit_rows || !r->val_same || !r->val_rows || !r->vslot) return -1;
    for (int i = 0; i < ntok; i++) r->vslot[i] = -1;
    for (int i = c->nfit; i < ntok; i++) r->vslot[c->perm[i]] = i - c->nfit;
    /* K 只有钩子给; val 留存按 MAXU 上限开(6 × 2048 × 40 很小), 选择分表 [nl][nval][E] */
    r->K = -1;
    r->vsc = malloc((size_t)nl * r->nval * r->E * 4); r->vself = malloc((size_t)nl * r->nval * MAXU * 4);
    if (!r->vsc || !r->vself) { fprintf(stderr, "★val 留存 %.0f MB 分配失败★\n", (double)nl * r->nval * r->E * 4 / 1e6); return -1; }
    return 0;
}
static void rb_seq_free(void) {
    rb_ctx *r = &g_rb;
    for (int il = 0; il < r->nl; il++) { if (r->fx[il]) fclose(r->fx[il]); cudaFree(r->dWg[il]); cudaFree(r->dB[il]); }
}

/* ★合并序贯(2026-09-20, 用户令"路由反修和正常反修合并重头跑")★: 第 k 遍的钩子同时干两件事 —— 路由统计(rb_hook)
 * 与增益取料(hook)。为什么不是各取一遍: 两者要的是同一块 x/sel, 同一趟拿到就是同一个部署态, 白跑一遍前向也换不来更准的账。
 * 只有 Δb 落盘之后本层路由变了, 增益才必须重取(驱动侧做, 见 v41_amp_run.c 的序贯循环)。 */
static int rbgr_hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel, const float *rw,
                     const float *alpha, const float *ye, const float *ysh) {
    ctx_t *c = ud;
    if (il != c->layer) return 0;
    if (rb_hook(ud, il, pos0, n, D, n_used, clamp, x, y, sel, rw, alpha, ye, ysh) < 0) return -1;
    return hook(ud, il, pos0, n, D, n_used, clamp, x, y, sel, rw, alpha, ye, ysh);
}
/* 合并序贯: 解第 il 层的 Δb(本遍取料已由 rbgr_hook 收进统计表)。返回 1 落盘 / 0 不挂 / <0 失败 */
static int rb_seq_solve(ctx_t *c, int il, int *nw) {
    rb_ctx *r = &g_rb;
    if (r->bad) return -1;
    if (!r->chk_rows[il]) { fprintf(stderr, "★L%02d 路由统计一行都没收到 —— 钩子没到这一层★\n", il); return -1; }
    return rb_solve_layer(r, il, c->mf, c->out_dir, nw);
}

/* 一趟前向拟合全部层的 Δb(部署态 = --base-amp 那份现役侧车, 路由统计不序贯)。返回 0 成功。 */
static int rb_fit(ctx_t *c, ds4_engine *e, const int *ids, int ntok, const char *lp, int no_engram, const char *dump, const char *base_amp, int nl) {
    if (rb_seq_init(c, ntok, nl)) return -1;
    rb_ctx *r = &g_rb;
    printf("[路由侧车] 部署态 %s; 拟合 %d / val %d 行; %d 层一趟统计; FP 路由 ← %s\n", base_amp ? base_amp : "(裸底座)", c->nfit, r->nval, nl, dump);
    ds4_engine_v41_set_amp_dir(base_amp);
    ds4_engine_v41_set_moe_hook(rb_hook, c);
    ds4_engine_v41_set_moe_hook_layer(-1);
    const double t0 = now_s();
    const int rc = ds4_engine_v41_score_ids(e, ids, ntok, lp, no_engram, 0);
    ds4_engine_v41_set_moe_hook(NULL, NULL);
    unlink(lp);
    if (rc != 0 || r->bad) { fprintf(stderr, "★拟合趟失败 rc=%d bad=%d★\n", rc, r->bad); return -1; }
    printf("[路由侧车] 取料+统计 %.0fs, route_scale %.4f\n", now_s() - t0, r->scale);
    int nw = 0;
    for (int il = 0; il < nl; il++) {
        if (!r->chk_rows[il]) { printf("[L%02d] 钩子没到这一层(取料 %ld 行), 跳过\n", il, r->chk_rows[il]); continue; }
        if (rb_solve_layer(r, il, c->mf, c->out_dir, &nw) < 0) return -1;
    }
    printf("[路由侧车] 落盘 %d 层 rb_Lnn.bin → %s\n", nw, c->out_dir);
    fprintf(c->mf, "# 完成 路由侧车 %d 层\n", nw);
    rb_seq_free();
    return 0;
}
