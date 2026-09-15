/* v41_margin_solve.inc.c — 后训练第三件的【解算与主动集】(2026-09-13 夜第三针)。
 * v41_amp_run.c 单 TU include; 取料与行表在 v41_sft_run.inc.c, 数值核在 v41_gr_solve.cu / v41_kl_target.cu。
 *
 * 【要解的是什么】部署是贪心 argmax: 位置 i 写哪个 token, 看的是谁的 logit 最大。所以
 *     "撬动一个错误判断" = 让对版 token 的 logit 压过★这一行榜上所有别人★。
 * 末层之后只有 norm→head, 所以"两个 token 的 logit 差"对本层 MoE 输出是精确线性的:
 *     Δ(ℓ_a − ℓ_b) = Σ_k Σ_d Δs[e_ik][d]·(rw_ik·ye_ik[d])·C[d],  C = α·inv·γ⊙(W[a]−W[b])
 * 于是每个"要 a 压过 b"就是一条线性方程, 一次共轭梯度解完。
 *
 * 【为什么要主动集 —— 09-13 实测逼出来的】第一版每行只钉一对(对版 vs 当时最强的那个对手)。
 * 真前向读数: 钉住的那一对★分毫不差★(预测 vs 真前向 相关 1.0000、一致 99.91%), 可是
 *   尺A 训练日决策点翻转 54/73 = 73.97%(预测说 100%)
 *   守门1 约束行 argmax 仍是原 top1 只有 2049/3268 = 62.70%(预测说"约束行动了 0.00%")
 * 两者不矛盾: 修正作用在整个 5120 维通道上, 会把一堆 token 的 logit 一起抬 —— 钉住的那对没变,
 * 冒头的是★第三个 token★。所以要钉的不是一对, 是一张榜。
 * 做法: 解完 → 用【线性预测】算出榜上每个 token 的 Δℓ(v41_klt_margin_scan, 毫秒级, 不跑真前向)
 * → 谁冒到了"该赢的那个"前面, 就把 (该赢的, 它) 加成新方程 → 重解。几轮就收敛。
 *
 * 【产物是 fp4】落盘是 fp4x32(4 bit/元素), 所以每一轮都是 解 → 量化 → 解回来 → 用【落地那一份】
 * 算 Δy 与扫榜。量化误差同样会让某个 token 冒头, 它和解算误差在这里被一视同仁地处理。 */

#include "../../src/common/ds4_gr_fnv.h"   /* 底座指纹: 与引擎核对用同一份算法 */

#define MG_ITER    200   /* CG 迭代上限: 1.97M 未知数但方程只有几千条, 实际几十步就到残差地板 */
#define MG_MAXCAND 64    /* τ×ρ×λ 网格的候选数上限 */
#define MG_ROUNDS  6     /* 主动集轮数上限 */
#define MG_PAIRMUL 4     /* 对数上限 = 行数 × 这个(方向表 C 按对存, 一对 D 个 f32) */

/* v41_gr_solve.cu 的决策差三个入口(实现与反修解算器同一个文件 —— 解算器只许有一份) */
int v41_gr_solve_margin_gpu(const float *dye, const float *drw, const int *dsel, const float *dC,
                            const int *dsrc, const float *db, const float *dw, int n, int nu, int D,
                            int n_expert, float lam, const float *dwe, const float *dgate, int R,
                            int niter, float *dx, float *dm, float *out_stat);
int v41_gr_margin_gram(const float *dye, const float *drw, const int *dsel, const float *dC,
                       const int *dsrc, int np, int nu, int D, float *dG);
int v41_gr_margin_predict(const float *dye, const float *drw, const int *dsel, const float *dC,
                          const int *dsrc, const float *dx, int npair, int nu, int D, int n_expert,
                          const float *dgate, int R, float *dm);
int v41_gr_margin_dy_gpu(const float *dye, const float *drw, const int *dsel, const float *dx,
                         int n, int nu, int D, int n_expert, const float *dgate, int R, float *dY);

/* 泛化侧在 v41_margin_gen.inc.c(它用本文件的 mg_candidate, 所以排在后面 include; 这里前置声明)。 */
static void mg_pair_rep(mg_acc *ab, int ndec_fit);
static float *mg_freq_we(ctx_t *c, mg_acc *ab, int ndec_fit, float kappa);
static float *mg_gates(ctx_t *c, mg_acc *ab, int ndec_fit, int R, float **hgate_out);
static int mg_diag(ctx_t *c, mg_acc *ab, int ndec_fit, int nsmp, const char *out, float *dC, int *dsrc_buf);

/* 读一个 gr_Lnn.bin(与引擎 core_v41_amp.c 的 v41_gr_accum_layer 同格式: 头 <ne><D><type>,
 * 之后是【偏离量 s−1】的 fp4x32)。返回 0 = 目录里没有这一层(按"不改"处理), 1 = 读到, <0 = 坏文件。
 * ★多晚累积靠它★: ③ₖ 落盘的必须是 s₃ₖ₋₁ ⊙ Δₖ —— 引擎只挂一个 ③ 目录, 只写 Δₖ 就把前几晚全丢了。 */
static int mg_read_gr(const char *dir, int il, int ne, int D, float *s_out) {
    if (!dir || !dir[0]) return 0;
    char p[4300]; snprintf(p, sizeof p, "%s/gr_L%02d.bin", dir, il);
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    int32_t hd[3];
    const size_t nel = (size_t)ne * D;
    if (fread(hd, 4, 3, f) != 3 || hd[0] != ne || hd[1] != D || hd[2] != (int32_t)DS4_GGT_FP4X32 || nel % 32u) {
        fprintf(stderr, "★%s 头不对(ne %d D %d type %d)★\n", p, hd[0], hd[1], hd[2]); fclose(f); return -1; }
    const size_t nblk = nel / 32u;
    uint8_t *pk = malloc(nblk * 17u);
    const int ok = pk && fread(pk, 1, nblk * 17u, f) == nblk * 17u;
    fclose(f);
    if (!ok) { free(pk); fprintf(stderr, "★%s 数据段截断★\n", p); return -1; }
    ds4_deq_fp4x32(pk, nblk, s_out);
    free(pk);
    for (size_t t = 0; t < nel; t++) s_out[t] += 1.0f;
    printf("[后训练] 上一版 ③ %s 读到 L%02d, 本轮产物 = 它 ⊙ 本轮解\n", dir, il);
    return 1;
}

/* ★底座指纹★: ③ 是解在 ①+②(+③上一版)之上的增量, 换了底座就不是这份修正了 —— 而错配照样能跑,
 * 出的还是个像模像样的读数。所以把 ② 目录的指纹写进 base.fnv, 引擎加载 ③ 时重算比对
 * (core_v41_amp.c: v41_pt_base_ok), 不符停车。 */
static int sft_write_base(ctx_t *c, const char *out) {
    unsigned nf = 0;
    const uint64_t fnv = (c->base_amp && c->base_amp[0]) ? ds4_gr_dir_fnv(c->base_amp, 64u, &nf) : DS4_GR_FNV_SEED;
    char bp[4300]; snprintf(bp, sizeof bp, "%s/base.fnv", out);
    FILE *bf = fopen(bp, "w");
    if (!bf) { fprintf(stderr, "★写不了 %s★\n", bp); return -1; }
    fprintf(bf, "%016llx %u\n", (unsigned long long)fnv, nf);
    if (fclose(bf)) { fprintf(stderr, "★%s 写盘失败★\n", bp); return -1; }
    return 0;
}

/* 方程集: 一个"对" = (取哪一行的料 src, 要 a 压过 b, 想推到多少 b_tgt, 权重 w)。
 * 首轮每行一对; 主动集每轮把"冒头的那个"追加成新对。 */
typedef struct {
    int   *src, *a, *b;
    float *m0, *bt, *w;
    int    n, cap;
} mg_pairs;

static int mg_pairs_alloc(mg_pairs *p, int cap) {
    memset(p, 0, sizeof *p);
    p->cap = cap;
    p->src = malloc((size_t)cap * 4); p->a = malloc((size_t)cap * 4); p->b = malloc((size_t)cap * 4);
    p->m0 = malloc((size_t)cap * 4); p->bt = malloc((size_t)cap * 4); p->w = malloc((size_t)cap * 4);
    if (!p->src || !p->a || !p->b || !p->m0 || !p->bt || !p->w) { fprintf(stderr, "★方程集分配失败★\n"); return -1; }
    return 0;
}
static void mg_pairs_free(mg_pairs *p) { free(p->src); free(p->a); free(p->b); free(p->m0); free(p->bt); free(p->w); }

typedef struct mg_cfg_s mg_cfg;   /* 定义在下面; 留一要按指针传 */
static int mg_loo(ctx_t *c, mg_acc *ab, mg_pairs *pp, int il, const char *out, mg_cfg *cf,
                  int ndec_fit, int ndec, int nsmp, float *dC, float *dm, const float *sprev,
                  float *dY, float *dLg, int *dIds, float *hLg, const unsigned char *base_win, float *stL);

static int mg_pair_add(mg_pairs *p, int src, int a, int b, float m0, float bt, float w) {
    for (int q = 0; q < p->n; q++)   /* 同一行同一对只进一次: 重复方程等于悄悄给它加权 */
        if (p->src[q] == src && p->a[q] == a && p->b[q] == b) return 0;
    if (p->n >= p->cap) return -1;
    const int i = p->n++;
    p->src[i] = src; p->a[i] = a; p->b[i] = b; p->m0[i] = m0; p->bt[i] = bt; p->w[i] = w;
    return 1;
}

/* 一个候选的全部旋钮。τ/ρ/λ 是第二版就有的; κ(频率岭)、ν(词对重复度)是第三版加的泛化先验,
 * hold_smp 是留一泛化尺(尺 L)——把某一条样本的决策行整条抽出方程之外, 只看它翻不翻。 */
struct mg_cfg_s {
    float tau, rho, lam, nuw;   /* nuw = ν: 决策方程权重 ×(1 + ν·(重复度−1)) */
    const float *dwe;           /* 专家频率岭 w_e(设备 [ne]); NULL = 全 1(旧行为) */
    int hold_smp;               /* ≥0 = 这条样本的决策行不进方程(折外, 只统计); −1 = 全量 */
    int no_write;               /* 1 = 不建目录不落盘(留一的每一折; fp4 往返照做) */
};

/* 一个候选。主动集循环: 解 → fp4 往返(落盘或不落) → 用落地那份算 Δy → 扫榜 → 谁冒头就加方程 → 重解。
 * st 回填: [0] 折内拟合决策点【argmax 口径】预测翻转率(%)  [1] 旧 val 段同上  [2] 约束行榜首保住率(%)
 *          [3] 落地 |s−1| 均值  [4] 最后一轮 CG 相对残差  [5] 方程数  [6] 主动集轮数
 *          [7] 收敛时仍压不服的行数(容量读数, 不是错误)  [8] 折外翻转数  [9] 折外总数 */
static int mg_candidate(ctx_t *c, mg_acc *ab, mg_pairs *pp, int il, const char *out,
                        const mg_cfg *cf, int ndec_fit, int ndec,
                        float *dC, float *dm, const float *sprev, float *dY, float *dLg, int *dIds,
                        float *hLg, const unsigned char *base_win, float *st) {
    const float tau = cf->tau, rho = cf->rho, lam = cf->lam;
    /* 折外行 = 留一时被抽出去的那条样本的决策行: 不进方程、主动集不管它, 只在统计里看翻没翻。 */
    #define MG_HELD(i) (cf->hold_smp >= 0 && (i) < ndec_fit && ab->smp[i] == cf->hold_smp)
    const int ns = ab->n, nu = ab->nu, D = ab->D, ne = c->n_expert, K = ab->K;
    /* ★共享模式(--share)★: 未知数只有 D 个, 全体专家共用一组通道增益。落盘时广播成 [ne][D]
     * (文件格式一个字节不改, 引擎照旧)。为什么: 见 v41_gr_solve.cu 里 est 那段的实测注释。 */
    const int nu_solve = c->share ? 1 : ne, R = c->gates > 0 ? c->gates : 1;
    const size_t nsD = (size_t)ne * D * (size_t)R, nxD = (size_t)nu_solve * D * (size_t)R;
    char dir[4300];
    snprintf(dir, sizeof dir, "%s/%s_g%d_t%g_r%g_l%g_k%g_n%g", out, c->share ? "candS" : "cand",
             R, tau, rho, lam, c->kappa, cf->nuw);
    if (!cf->no_write && mkdir(dir, 0775) && errno != EEXIST) { fprintf(stderr, "★建不了 %s★\n", dir); return -1; }

    /* 首轮方程: 每行一对。决策拟合行 w=1 且要推到 τ; 决策 val 行 w=0(解算一行不看, 只观察);
     * 约束行 w=ρ 且 b=0(= "这里原本最想说什么, 别给我改")。 */
    pp->n = 0;
    for (int i = 0; i < ns; i++) {
        const float w = MG_HELD(i) ? 0.f
                      : (i < ndec_fit) ? 1.f + cf->nuw * (ab->repw[i] - 1.f)
                      : (i < ndec ? 0.f : rho);
        const float bt = (i < ndec_fit) ? tau - ab->m0[i] : 0.f;
        if (mg_pair_add(pp, i, ab->ta[i], ab->tb[i], ab->m0[i], bt, w) < 0) {
            fprintf(stderr, "★方程集首轮就超上限 %d★\n", pp->cap); return -1; }
    }
    int *dsrc = NULL; float *db = NULL, *dw = NULL;
    if (cudaMalloc((void **)&dsrc, (size_t)pp->cap * 4) || cudaMalloc((void **)&db, (size_t)pp->cap * 4) ||
        cudaMalloc((void **)&dw, (size_t)pp->cap * 4)) { fprintf(stderr, "★候选设备缓冲失败★\n"); return -1; }
    int rc = -1, built = 0, round = 0, left_d = 0, left_c = 0;
    for (; round < MG_ROUNDS; round++) {
        rc = -1;   /* 每轮从"没成"起算: 中途任何一步 break 出去都必须是失败 */
        /* 方向表只给新加的对建(旧的不变) —— 一对一个 5120 维向量, 重建整张表是白烧 */
        if (pp->n > built) {
            const int nnew = pp->n - built;
            float *al = malloc((size_t)nnew * 4), *iv = malloc((size_t)nnew * 4);
            if (!al || !iv) { free(al); free(iv); break; }
            for (int q = 0; q < nnew; q++) { const int r = pp->src[built + q]; al[q] = ab->alpha[r]; iv[q] = ab->inv[r]; }
            const int drc = v41_klt_margin_dirs(c->klt, pp->a + built, pp->b + built, al, iv, nnew, D,
                                                dC + (size_t)built * D);
            free(al); free(iv);
            if (drc) break;
            built = pp->n;
        }
        float cg[2] = { 0.f, 0.f };
        if (cudaMemcpy(dsrc, pp->src, (size_t)pp->n * 4, cudaMemcpyHostToDevice) ||
            cudaMemcpy(db, pp->bt, (size_t)pp->n * 4, cudaMemcpyHostToDevice) ||
            cudaMemcpy(dw, pp->w, (size_t)pp->n * 4, cudaMemcpyHostToDevice)) break;
        if (v41_gr_solve_margin_gpu(c->dye, c->drw, c->dsel, dC, dsrc, db, dw, pp->n, nu, D, nu_solve,
                                    lam, c->share ? NULL : cf->dwe, c->dgate, R, MG_ITER, c->dgr, dm, cg)) break;
        st[4] = cg[0];
        /* ★fp4 往返必须在扫榜之前★: 落盘是 4 bit/元素, 解算态的 Δ 与落地的不是一个东西。
         * 落的还是【与上一版 ③ 相乘之后】的 s, 所以"有效 Δ" = 量化后的乘积 ÷ 上一版 − 1。 */
        if (cudaMemcpy(c->hgr, c->dgr, nxD * 4, cudaMemcpyDeviceToHost)) break;
        /* 共享模式: 先把 D 个数广播成 [ne][D], 再与上一版相乘落盘 —— 量化误差必须落在
         * 【落地那一份】上, 所以广播在量化之前做。 */
        if (c->share)
            for (int r = R - 1; r >= 0; r--)
                for (size_t e = ne; e-- > 0; )
                    for (int d = 0; d < D; d++)
                        c->hgr[((size_t)r * ne + e) * D + d] = c->hgr[(size_t)r * (size_t)D + d];
        for (size_t t = 0; t < nsD; t++) c->hgr[t] = (1.0f + c->hgr[t]) * sprev[t % ((size_t)ne * D)];
        if (R > 1) {   /* ★x 键控是验证形态★: 引擎侧还没有它的载体(要加一个小算子), 所以只往返不落盘 */
            uint8_t *pk = gr_fp4_roundtrip(c->hgr, nsD);
            if (!pk) break;
            free(pk);
        } else if (cf->no_write) {   /* 留一: 同一份 fp4 往返(尺 L 与 gate 必须量同一个态), 只是不落盘 */
            uint8_t *pk = gr_fp4_roundtrip(c->hgr, nsD);
            if (!pk) break;
            free(pk);
        } else if (gr_write_bin_roundtrip(c->hgr, ne, D, dir, il)) break;
        for (size_t t = 0; t < nsD; t++) {
            const float sp = sprev[t % ((size_t)ne * D)];
            c->hgr[t] = sp != 0.f ? c->hgr[t] / sp - 1.0f : 0.f;
        }
        /* 回读之后不再共享(fp4 是逐 32 元素一个 scale, 广播后各专家的量化误差不同) ⇒
         * 扫榜一律按 [ne][D] 走, 量的就是落地那一份的真实效果。 */
        if (cudaMemcpy(c->dgrf, c->hgr, nsD * 4, cudaMemcpyHostToDevice)) break;
        /* 扫榜: 本层输出被挪了多少 → 榜上每个 token 的 logit 各动了多少 */
        if (v41_gr_margin_dy_gpu(c->dye, c->drw, c->dsel, c->dgrf, ns, nu, D, ne, c->dgate, R, dY)) break;
        if (v41_klt_margin_scan(c->klt, dY, ab->alpha, ab->inv, dIds, ns, K, D, dLg)) break;
        if (cudaMemcpy(hLg, dLg, (size_t)ns * K * 4, cudaMemcpyDeviceToHost)) break;
        rc = 0;
        /* 谁冒到了"该赢的那个"前面 —— 拟合决策行与约束行都要管(val 行只观察, 不加方程) */
        int add = 0, viol_d = 0, viol_c = 0;
        for (int i = 0; i < ns; i++) {
            if ((i >= ndec_fit && i < ndec) || MG_HELD(i)) continue;
            float wl = ab->wlp[i];
            for (int q = 0; q < K; q++)   /* "该赢的"若在榜上, 它自己的 Δℓ 也要加 */
                if (ab->tid[(size_t)i * K + q] == ab->want[i]) { wl += hLg[(size_t)i * K + q]; break; }
            int bq = -1; float bl = -1e30f;
            for (int q = 0; q < K; q++) {
                if (ab->tid[(size_t)i * K + q] < 0 || ab->tid[(size_t)i * K + q] == ab->want[i]) continue;
                const float lv = ab->tlp[(size_t)i * K + q] + hLg[(size_t)i * K + q];
                if (lv > bl) { bl = lv; bq = q; }
            }
            if (bq < 0 || bl <= wl) continue;                      /* 没人冒头 */
            if (i < ndec_fit) viol_d++; else viol_c++;
            /* 新方程的 m⁰ 必须是【修正之前】的差: 每一轮都是从 x=0 重解整套方程, 不是接着上一轮改 */
            const float m0 = ab->wlp[i] - ab->tlp[(size_t)i * K + bq];
            const float bt = (i < ndec_fit) ? tau - m0 : 0.f;
            const float w  = (i < ndec_fit) ? 1.f + cf->nuw * (ab->repw[i] - 1.f) : rho;
            const int ar = mg_pair_add(pp, i, ab->want[i], ab->tid[(size_t)i * K + bq], m0, bt, w);
            if (ar < 0) { fprintf(stderr, "  ★方程数到上限 %d, 主动集停在第 %d 轮★\n", pp->cap, round + 1); break; }
            add += ar;
        }
        left_d = viol_d; left_c = viol_c;
        if (!add) { round++; break; }
        printf("    主动集第 %d 轮: 冒头 决策行 %d / 约束行 %d → 加 %d 条方程(共 %d)\n",
               round + 1, viol_d, viol_c, add, pp->n);
        fflush(stdout);
        /* ★跑满轮数不算失败★(09-13 实撞: 原来这里置 rc=-1, 于是 12 个候选里 11 个有效产物被当成
         * 失败丢掉)。最后一轮解出来的东西是完整的, 只是还有零星几行没压服 —— 那是【容量读数】,
         * 该报出来让人看见, 不是错误。 */
    }
    if (rc == 0) {
        /* 统计全部按★argmax 口径★(与真前向的尺 A / 守门 1 同一个量), 不再看单对的 m>0 */
        int f_ok = 0, v_ok = 0, c_keep = 0, h_ok = 0, h_n = 0, f_n = 0;
        int h_broke = 0, h_broke_trained = 0;   /* 折外"基线对→解后错"的点; 其中新 argmax 是训练里被推高过的 token */
        double sabs = 0.0;
        for (size_t t = 0; t < nsD; t++) sabs += fabs((double)c->hgr[t]);
        for (int i = 0; i < ns; i++) {
            float wl = ab->wlp[i], bl = -1e30f;
            for (int q = 0; q < K; q++) {
                const int v = ab->tid[(size_t)i * K + q];
                if (v < 0) continue;
                const float lv = ab->tlp[(size_t)i * K + q] + hLg[(size_t)i * K + q];
                if (v == ab->want[i]) wl = lv;
                else if (lv > bl) bl = lv;
            }
            const int win = (wl > bl);
            if (MG_HELD(i)) {
                h_ok += win; h_n++;
                /* ★全局词偏置假说的直接证据★: 折外点原来是对的, 解完变错了, 抢走它的是谁?
                 * 若抢走它的 token 恰恰是训练决策点里"我们推高过"的那些(对版 token 集合),
                 * 就说明这个形态学到的不是"什么情况下该说什么", 而是"无条件偏爱某几个词"。 */
                if (!win && base_win[i]) {
                    h_broke++;
                    int newtok = -1; float nb = -1e30f;
                    for (int q = 0; q < K; q++) {
                        const int v = ab->tid[(size_t)i * K + q];
                        if (v < 0 || v == ab->want[i]) continue;
                        const float lv = ab->tlp[(size_t)i * K + q] + hLg[(size_t)i * K + q];
                        if (lv > nb) { nb = lv; newtok = v; }
                    }
                    for (int j = 0; j < ndec_fit; j++)
                        if (!MG_HELD(j) && ab->want[j] == newtok) { h_broke_trained++; break; }
                }
            }
            else if (i < ndec_fit) { f_ok += win; f_n++; }
            else if (i < ndec) v_ok += win;
            else c_keep += win;
        }
        st[8] = (float)h_ok; st[9] = (float)h_n;
        st[10] = (float)h_broke; st[11] = (float)h_broke_trained;
        st[0] = f_n ? 100.f * f_ok / f_n : 0.f;
        st[1] = (ndec > ndec_fit) ? 100.f * v_ok / (ndec - ndec_fit) : 0.f;
        st[2] = (ns > ndec) ? 100.f * c_keep / (ns - ndec) : 0.f;
        st[3] = (float)(sabs / (double)nsD);
        st[5] = (float)pp->n; st[6] = (float)round; st[7] = (float)(left_d + left_c);
        /* predict.txt: 自检 2 用(挂上候选重打一趟分, 拿新表算真 m 与这里的 m_pred 对) */
        char pq[4300]; snprintf(pq, sizeof pq, "%s/predict.txt", dir);
        FILE *pf = cf->no_write ? NULL : fopen(pq, "w");
        if (pf) {
            float *hm = malloc((size_t)pp->n * 4);
            if (hm && !cudaMemcpy(hm, dm, (size_t)pp->n * 4, cudaMemcpyDeviceToHost)) {
                fprintf(pf, "# 段 样本 行号 token_a token_b m0 dm m_pred  (段: dec_fit/dec_val/ctr; m>0 = 贪心写 a)\n");
                for (int q = 0; q < pp->n; q++) {
                    const int i = pp->src[q];
                    fprintf(pf, "%s %d %d %d %d %.6f %.6f %.6f\n",
                            i < ndec_fit ? "dec_fit" : (i < ndec ? "dec_val" : "ctr"),
                            ab->smp[i], ab->rowid[i], pp->a[q], pp->b[q], pp->m0[q], hm[q], pp->m0[q] + hm[q]);
                }
            }
            free(hm);
            fclose(pf);
        }
        if (!cf->no_write && sft_write_base(c, dir)) rc = -1;
    }
    cudaFree(dsrc); cudaFree(db); cudaFree(dw);
    return rc;
    #undef MG_HELD
}

/* 逗号分隔的浮点表 → 数组(τ/ρ/λ 网格)。 */
static int mg_grid(const char *s, float *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p && n < cap) {
        char *end = NULL;
        const float v = strtof(p, &end);
        if (end == p) return 0;
        out[n++] = v;
        p = end;
        if (*p == ',') p++;
        else if (*p) return 0;
    }
    return n;
}

/* τ×ρ×λ 网格: 一次取料出多个候选。上传行料 → 逐候选(解+主动集) → 候选表按预测排序。
 * ★这里出的全是【预测】★, 真前向的尺 A/尺 B 与三把守门尺在夜间脚本的 gate/gatea 段。
 * 预测与真前向对不上本身就是要查的东西(back.md §4.5 自检 2): 那说明取料行号、α、inv、W 行、
 * bf16 五处之一错位, 不是"方法不行"。 */
static int mg_solve_grid(ctx_t *c, mg_acc *ab, int il, const char *out, int ndec_fit, int ndec) {
    const int ns = ab->n, nu = ab->nu, D = ab->D, K = ab->K;
    /* ★R 网格★: 一次取料把 R=1(旧形态, 自检基准)与 R>1(x 键控)一起跑完 —— 取料 8~20 分钟,
     * 解算秒级, 分两次发车等于白烧一趟前向。 */
    float gts[MG_MAXCAND];
    const int ng = mg_grid(c->gate_s, gts, MG_MAXCAND);
    int maxR = 1;
    for (int q = 0; q < ng; q++) if ((int)gts[q] > maxR) maxR = (int)gts[q];
    if (!ng) { fprintf(stderr, "★--gates %s 不合法★\n", c->gate_s); return -1; }
    const size_t nyD = (size_t)ns * nu * D, nsD = (size_t)c->n_expert * D * (size_t)maxR;
    const int pcap = ns * MG_PAIRMUL;
    cudaFree(c->dye); cudaFree(c->drw); cudaFree(c->dsel);
    cudaFree(c->dysh); cudaFree(c->dX); cudaFree(c->dYq); cudaFree(c->dYfp); cudaFree(c->dYtmp);
    c->dysh = c->dX = c->dYq = c->dYfp = c->dYtmp = NULL;   /* 决策差路一个都不用: 腾给 ye 与方向表 */
    float *dC = NULL, *dm = NULL, *dY = NULL, *dLg = NULL; int *dIds = NULL;
    if (cudaMalloc((void **)&c->dye, nyD * 4) || cudaMalloc((void **)&c->drw, (size_t)ns * nu * 4) ||
        cudaMalloc((void **)&c->dsel, (size_t)ns * nu * 4) || cudaMalloc((void **)&dC, (size_t)pcap * D * 4) ||
        cudaMalloc((void **)&dm, (size_t)pcap * 4) || cudaMalloc((void **)&dY, (size_t)ns * D * 4) ||
        cudaMalloc((void **)&dLg, (size_t)ns * K * 4) || cudaMalloc((void **)&dIds, (size_t)ns * K * 4)) {
        fprintf(stderr, "★设备缓冲(ye %.2f GB + 方向表 %.2f GB)分配失败★\n",
                (double)nyD * 4 / 1e9, (double)pcap * D * 4 / 1e9); return -1; }
    CK(cudaMemcpy(c->dye, ab->ye, nyD * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->drw, ab->rw, (size_t)ns * nu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dsel, ab->sel, (size_t)ns * nu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dIds, ab->tid, (size_t)ns * K * 4, cudaMemcpyHostToDevice));
    /* ★按 maxR 重配增益缓冲★: main 在 --capture-ye 时已经按"一份"(反修路口径 [384][5120])
     * 预分配过 dgr/hgr, 沿用它的话 x 键控的 R 份会安静越界 —— 09-14 实撞: R=4 时
     * cudaMemset 报 invalid argument, 而报错行指向的是下游第一处 CUDA 调用, 完全指错地方。 */
    cudaFree(c->dgr); c->dgr = NULL;
    free(c->hgr); c->hgr = malloc(nsD * 4);
    if (!c->hgr) { fprintf(stderr, "★增益主机缓冲 %.1f MB 分配失败★\n", (double)nsD * 4 / 1e6); return -1; }
    if (cudaMalloc((void **)&c->dgr, nsD * 4)) { fprintf(stderr, "★增益缓冲分配失败★\n"); return -1; }
    if (!c->dgrf && cudaMalloc((void **)&c->dgrf, nsD * 4)) { fprintf(stderr, "★落地态增益缓冲分配失败★\n"); return -1; }
    float *hLg = malloc((size_t)ns * K * 4);
    /* 上一版 ③ 的缩放因子(没有就是全 1): 本轮产物 = 它 ⊙ 本轮解, 见 mg_read_gr */
    float *sprev = malloc((size_t)c->n_expert * D * 4);
    mg_pairs pp;
    if (!hLg || !sprev || mg_pairs_alloc(&pp, pcap)) { free(hLg); free(sprev); return -1; }
    for (size_t t = 0; t < (size_t)c->n_expert * D; t++) sprev[t] = 1.0f;
    if (mg_read_gr(c->base_pt, il, c->n_expert, D, sprev) < 0) { free(hLg); free(sprev); mg_pairs_free(&pp); return -1; }

    /* 解算前的自证: 基线 argmax 口径的翻转率, 应当与判决器 --ref-topk 在同一批行上报的一致
     * (同一张 top-K 表、同一套行号)。差得多 = 行号或 tgt 口径错位, 后面全是假账。 */
    float base_fit = 0.f;   /* 基线折内拟合翻转率 = 尺 L 的比较基准(同一批点, 解前解后) */
    unsigned char *base_win = calloc((size_t)ns, 1);   /* 每行基线 argmax 对不对: 诊断"谁抢走了折外点"要它 */
    if (!base_win) { free(hLg); free(sprev); mg_pairs_free(&pp); return -1; }
    int b_f = 0, b_v = 0, b_c = 0;
    for (int i = 0; i < ns; i++) {
        float wl = ab->wlp[i], bl = -1e30f;
        for (int q = 0; q < K; q++) {
            const int v = ab->tid[(size_t)i * K + q];
            if (v < 0) continue;
            const float lv = ab->tlp[(size_t)i * K + q];
            if (v == ab->want[i]) wl = lv; else if (lv > bl) bl = lv;
        }
        const int win = (wl > bl);
        base_win[i] = (unsigned char)win;
        if (i < ndec_fit) b_f += win; else if (i < ndec) b_v += win; else b_c += win;
    }
    base_fit = ndec_fit ? 100.f * b_f / ndec_fit : 0.f;
    printf("[后训练·基线 argmax 口径] 决策点 拟合 %d/%d = %.2f%% / val %d/%d = %.2f%%; 约束行榜首 %d/%d = %.2f%%\n",
           b_f, ndec_fit, ndec_fit ? 100.0 * b_f / ndec_fit : 0.0,
           b_v, ndec - ndec_fit, (ndec > ndec_fit) ? 100.0 * b_v / (ndec - ndec_fit) : 0.0,
           b_c, ns - ndec, (ns > ndec) ? 100.0 * b_c / (ns - ndec) : 0.0);

    float taus[MG_MAXCAND], rhos[MG_MAXCAND], lams[MG_MAXCAND], kaps[MG_MAXCAND], nus[MG_MAXCAND];
    const int nt = mg_grid(c->tau_s, taus, MG_MAXCAND), nr = mg_grid(c->rho_s, rhos, MG_MAXCAND),
              nl = mg_grid(c->lam_s, lams, MG_MAXCAND), nk = mg_grid(c->kap_s, kaps, MG_MAXCAND),
              nn = mg_grid(c->nu_s, nus, MG_MAXCAND);
    if (!nt || !nr || !nl || !nk || !nn || nt * nr * nl * nk * nn > MG_MAXCAND) {
        fprintf(stderr, "★网格 τ=%s ρ=%s λ=%s κ=%s ν=%s 不合法或候选数 %d > %d★\n",
                c->tau_s, c->rho_s, c->lam_s, c->kap_s, c->nu_s, nt * nr * nl * nk * nn, MG_MAXCAND);
        free(hLg); free(sprev); mg_pairs_free(&pp); return -1; }
    /* ★词对重复度★(先算: 它进决策方程的权重) 与 ★诊断★(泛化上限的刻度, 秒级, 先出数再烧网格) */
    mg_pair_rep(ab, ndec_fit);
    {
        int *dsrc_d = NULL;
        if (!cudaMalloc((void **)&dsrc_d, (size_t)ndec_fit * 4)) {
            if (mg_diag(c, ab, ndec_fit, c->nsmp, out, dC, dsrc_d)) fprintf(stderr, "★诊断失败(不停车, 继续解)★\n");
            cudaFree(dsrc_d);
        }
    }
    char cp[4300]; snprintf(cp, sizeof cp, "%s/candidates.txt", out);
    FILE *cf = fopen(cp, "w");
    if (!cf) { fprintf(stderr, "★候选表写不了 %s★\n", cp); free(hLg); free(sprev); mg_pairs_free(&pp); return -1; }
    fprintf(cf, "# 目录 R tau rho lam kappa nu ★尺L折外翻转%%★ 拟合决策点翻转%% val决策点翻转%% 约束行榜首保住%% |s-1|均值 CG残差 方程数 主动集轮数 残余冒头\n"
                "# ★全是【预测】★(落地 fp4 态、argmax 口径), 真前向判决在夜间脚本 gate 段。\n"
                "# ★排序键 = 尺 L(按样本条留一的折外翻转率)★, 不是拟合率 —— 按拟合率排就是专挑最会背题的\n"
                "#   那个候选(09-14 实测: 拟合 76.92%% 与 57.26%% 的两个候选, 判决点翻转一模一样)。\n"
                "# 基线: 拟合 %.2f%% / val %.2f%% / 约束行榜首 %.2f%%\n",
            ndec_fit ? 100.0 * b_f / ndec_fit : 0.0,
            (ndec > ndec_fit) ? 100.0 * b_v / (ndec - ndec_fit) : 0.0,
            (ns > ndec) ? 100.0 * b_c / (ns - ndec) : 0.0);
    int nok = 0;
    for (int gq = 0; gq < ng; gq++) {
      c->gates = (int)gts[gq];
      if (c->dgate) { cudaFree(c->dgate); c->dgate = NULL; }
      /* ★门控★(R=1 只有恒等门 ⇒ 与旧形态逐位相同, 这就是新代码的自检基准) */
      c->dgate = mg_gates(c, ab, ndec_fit, c->gates, NULL);
      if (!c->dgate) break;
      printf("[后训练·形态] R=%d 门(未知数 %d × 384 × 5120)\n", c->gates, c->gates);
      for (int kq = 0; kq < nk; kq++) {
        c->kappa = kaps[kq];
        float *dwe = mg_freq_we(c, ab, ndec_fit, kaps[kq]);
        if (!dwe) break;
        for (int nq = 0; nq < nn; nq++)
        for (int a = 0; a < nt; a++) for (int b = 0; b < nr; b++) for (int g = 0; g < nl; g++) {
            float st[16] = { 0 }, stL[8] = { 0 };
            const double t0 = now_s();
            mg_cfg cfg; memset(&cfg, 0, sizeof cfg);
            cfg.tau = taus[a]; cfg.rho = rhos[b]; cfg.lam = lams[g]; cfg.nuw = nus[nq];
            cfg.dwe = dwe; cfg.hold_smp = -1; cfg.no_write = 0;
            if (mg_candidate(c, ab, &pp, il, out, &cfg, ndec_fit, ndec,
                             dC, dm, sprev, dY, dLg, dIds, hLg, base_win, st)) {
                fprintf(stderr, "★候选 τ=%g ρ=%g λ=%g κ=%g ν=%g 解算失败★\n",
                        taus[a], rhos[b], lams[g], kaps[kq], nus[nq]); continue; }
            /* ★尺 L★: 同一组旋钮再按样本条留一解一遍(每折不落盘)。它才是候选的排序键。 */
            if (mg_loo(c, ab, &pp, il, out, &cfg, ndec_fit, ndec, c->nsmp,
                       dC, dm, sprev, dY, dLg, dIds, hLg, base_win, stL)) {
                fprintf(stderr, "★留一失败, 停车★\n"); fclose(cf); free(hLg); free(sprev); free(base_win);
                mg_pairs_free(&pp); cudaFree(dwe); return -1; }
            printf("  [R=%d τ=%g ρ=%g λ=%g κ=%g ν=%g] ★尺L %.2f%%(基线 %.2f%%, Δ %+.2fpp; 折外 %d 点, %.0fs)★"
                   " 折外被弄坏 %d 点(其中 %d 个被★训练里推高过的 token★抢走) | 拟合 %.1f%% / val %.1f%% | "
                   "约束行榜首保住 %.2f%% | |s−1| 均 %.4f | 方程 %d 轮 %d 残余冒头 %d (%.1fs)\n",
                   c->gates, taus[a], rhos[b], lams[g], kaps[kq], nus[nq], stL[0], base_fit, stL[0] - base_fit, (int)stL[1], stL[2],
                   (int)stL[3], (int)stL[4],
                   st[0], st[1], st[2], st[3], (int)st[5], (int)st[6], (int)st[7], now_s() - t0);
            fprintf(cf, "%s_g%d_t%g_r%g_l%g_k%g_n%g %d %g %g %g %g %g %.2f %.2f %.2f %.2f %.5f %.3e %d %d %d\n",
                    c->share ? "candS" : "cand", c->gates, taus[a], rhos[b], lams[g], kaps[kq], nus[nq],
                    c->gates, taus[a], rhos[b], lams[g], kaps[kq], nus[nq], stL[0], st[0], st[1], st[2], st[3], st[4],
                    (int)st[5], (int)st[6], (int)st[7]);
            fflush(cf); nok++;
        }
        cudaFree(dwe);
      }
    }
    fclose(cf); free(hLg); free(sprev); free(base_win); mg_pairs_free(&pp);
    cudaFree(dC); cudaFree(dm); cudaFree(dY); cudaFree(dLg); cudaFree(dIds);
    printf("[后训练] %d 个候选落在 %s; 挑哪个由 gate 段真前向定\n", nok, out);
    return nok ? 0 : -1;
}
