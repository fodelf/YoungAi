/* dsq_units_sweep.inc.c — ③ sweep 单元的候选评估(物理分片, 只被 dsq_units.c include)。
 * 拆出来的唯一原因是 500 行守卫; 逻辑上它就是 dsq_units.c 的 ③ 段, 不是另一份实现。
 * 必须在 ds4_z.c / ds4quant_fwd.c / ds4quant_zsolve.inc.c 之后 include。 */
/* ★③ sweep 的真活在这里★(2026-08-28)
 * bf_exit_relL2 住在 ds4quant_bfkernel.inc.c, 它依赖 HCM/DIM 宏(生产 p1:174)与
 * g_anc_rowmap/g_anc_rowstride(生产 p6:360)。单元里锚行恒等(不抽格) ⇒ rowmap=NULL。 */
#define HCM 4
static const int *g_anc_rowmap    = NULL;   /* 抽格行→原始行; NULL=恒等, 同生产 */
static int        g_anc_rowstride = 0;
#include "ds4quant_bfkernel.inc.c"

/* (c) ★候选评估★ —— 生产 ds4quant_run_p7.inc.c:410-440(ZLGATE 段)逐行搬出。
 * 单层 sweep 的成本【全部】在这儿: 每个候选一次全 S 行 hc 合成(dq_hc_post) +
 * 一次 val 行出口分(bf_exit_relL2)。上面 (a) 那句择优比较是 0 成本的收尾。
 * ★我最初只搬了 (a), 所以 ③ 报 0.000s —— 那不是"快", 是根本没测东西。★
 * 首个过门的候选【立即落地并写回 Fout】(序贯: 下一层看到校正后的激活), 与生产同,
 * 不是"全扫完再挑最优" —— 改成后者会与生产的落地序偏离, 结果不可比。
 * 信任域夹持 ‖z(x_s)‖ ≤ LZTR·‖routed_s‖ 防把激活拉出流形(生产 LZTR=0.5;
 * 2026-08-31 口径正修: 基准=routed, 与 zreplay/引擎同; shb=共享基, NULL=Fout 已是 routed)。
 * 返回 1=有候选落地 0=全拒(Fout 不动), -1=内存失败。 */
int dsq_sweep_layer(ds4_z *zl, const float *Fin, float *Fout, float *Ftry,
                    const float *resid, const float *post, const float *comb,
                    float *Hq, const float *Hf, const float *shb,
                    int S, int vs, int n_fit, float LZTR,
                    const int *cand, int ncand,
                    int *k_land, double *e0_out, double *e1_out, int *n_eval)
{
    /* 基线 val 出口 relL2(未加任何校正的 Fout) */
    dq_hc_post(Fout, resid, post, comb, Hq, S, HCM, DIM);
    const double e0 = bf_exit_relL2(Hq, Hf, vs, n_fit);
    if (e0_out) *e0_out = e0;

    float *zd = malloc((size_t)DIM * 4);
    int   *tried = calloc((size_t)(ncand > 0 ? ncand : 1), sizeof(int));
    if (!zd || !tried) { free(zd); free(tried); return -1; }

    int landed = 0, ne = 0;
    for (int ci = 0; ci < ncand && !landed; ci++) {
        const int kk = cand[ci];
        if (kk < 1 || kk > (int)zl->rank) continue;          /* 越界候选: 不算评估 */
        int dup = 0;
        for (int cj = 0; cj < ci; cj++) if (cand[cj] == kk && tried[cj]) dup = 1;
        if (dup) continue;                                   /* 生产同款去重 */
        tried[ci] = 1; ne++;

        ds4_z_set_rank(zl, (uint32_t)kk);
        memcpy(Ftry, Fout, (size_t)S * DIM * 4);
        for (int sx = 0; sx < S; sx++) {
            memset(zd, 0, (size_t)DIM * 4);
            ds4_z_apply(zl, Fin + (size_t)sx * DIM, zd);
            double nd = 0, nf = 0;
            const float *fo = Ftry + (size_t)sx * DIM;
            const float *zb = shb ? shb + (size_t)sx * DIM : NULL;
            for (int d = 0; d < DIM; d++) { nd += (double)zd[d]*zd[d];
                const double rt = (double)fo[d] - (zb ? zb[d] : 0.0); nf += rt*rt; }
            nd = sqrt(nd); nf = sqrt(nf);
            const double cap = (double)LZTR * nf;
            float sc = 1.0f;
            if (nd > cap && nd > 0) sc = (float)(cap / nd);   /* 信任域夹持 */
            float *fw = Ftry + (size_t)sx * DIM;
            for (int d = 0; d < DIM; d++) fw[d] += sc * zd[d];
        }
        dq_hc_post(Ftry, resid, post, comb, Hq, S, HCM, DIM);
        const double e1 = bf_exit_relL2(Hq, Hf, vs, n_fit);
        printf("    ZLGATE k=%-4d val出口relL2 %.6f→%.6f %s\n",
               kk, e0, e1, e1 < e0 - 1e-9 ? "✓落地" : "✗拒");
        fflush(stdout);
        if (e1 < e0 - 1e-9) {
            landed = 1;
            if (k_land) *k_land = kk;
            if (e1_out) *e1_out = e1;
            memcpy(Fout, Ftry, (size_t)S * DIM * 4);          /* 序贯写回 */
        }
    }
    free(zd); free(tried);
    if (n_eval) *n_eval = ne;
    return landed;
}
