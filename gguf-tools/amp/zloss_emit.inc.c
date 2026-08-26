/* zloss_emit.inc.c — GE-only 端到端注入(斜率标定, 只被 zloss_solve.c include)。
 *
 * 依据(2026-08-26 针终判+用户令"跑"): per-layer held 修正家族饱和 ~1%/层, 但
 * held层内≠端到端(冠军 43 层小赢复利出 KL−9.7%)——唯一裁判=caliper。本分片把
 * zloss 家族唯一稳定赢家 GE(每专家门)落成冻结判决尺原生认的 bf.GE 记录, 43 层
 * 注入后跑 caliper_ref.sh, 标定 held↔端到端换算斜率。
 *
 * 记录=zlayer make_rec 同构(116B 头: 名@0/psz u64@88/vd=1@112, 载荷 256×f16 增益);
 * 判决尺语义: 专家累加时乘 g_e(ds4quant_run_p4), 学生基线 Ys=Σw·y ⇒ g_e=1+δ_e
 * (solve_ge 解的是 R≈Σδ_e·w·y 的 δ)。账本 zinject_manifest.txt 同 zlayer
 * (每行"L osz n0", 回滚=按账截断+回写 nrec)。
 * 闸=GE held 四损失 total 严格赢裸才注入(zlayer GATE 同语义); 输裸层零动作,
 * 不算停车审计(斜率标定要的就是"哪些层赢、赢多少、合起来值多少"这张表)。
 */

#include <sys/stat.h>
#include "src/common/ds4_float.h"

static void emitge_inject(const char *ldir, int L, const float *g256) {
    char dql[1200], man[1200];
    snprintf(dql, sizeof dql, "%s/dql_L%02d.bin", ldir, L);
    snprintf(man, sizeof man, "%s/zinject_manifest.txt", ldir);
    FILE *mf = fopen(man, "r");     /* 全量重跑铁律: 干净工作区不该有本层账 */
    if (mf) {
        char line[256];
        while (fgets(line, sizeof line, mf))
            if (atoi(line) == L && strchr(line, ' '))
                die("L%d 已在注入账本 %s — 工作区不是干净态, 停车", L, man);
        fclose(mf);
    }
    uint8_t rec[116 + 512];
    memset(rec, 0, sizeof rec);
    memcpy(rec, "bf.GE", 5);
    uint64_t psz = 512; memcpy(rec + 88, &psz, 8);
    int32_t vd = 1; memcpy(rec + 112, &vd, 4);
    uint16_t *g16 = (uint16_t *)(rec + 116);
    for (int e = 0; e < 256; e++) g16[e] = ds4_f64_to_f16((double)g256[e]);
    struct stat ds;
    if (stat(dql, &ds)) die("dql 不存在: %s", dql);
    long long osz = (long long)ds.st_size;
    FILE *df = fopen(dql, "r+b");
    if (!df) die("dql 打不开(r+b): %s", dql);
    uint32_t n0;
    if (fseeko(df, 8, SEEK_SET) || fread(&n0, 4, 1, df) != 1) die("dql nrec 读不到");
    if (fseeko(df, 0, SEEK_END) || fwrite(rec, 1, sizeof rec, df) != sizeof rec)
        die("dql 追加失败");
    uint32_t n1 = n0 + 1;
    if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
    fclose(df);
    mf = fopen(man, "a");
    if (!mf) die("账本写不开: %s", man);
    fprintf(mf, "%d %lld %u\n", L, osz, n0);
    fclose(mf);
}

/* GE-only 一层: 解算(λ=1e-3 zlayer GE_LAM 同款)→held 四损失判→过闸注入。
 * 打印行与针的 GE 臂同源(run_bc_arm), 数字口径与三层针逐位可比。 */
static void run_emit_ge(const char *ldir, int L, const zpairs *zp, const float *X,
                        const float *R, const float *Ys, const float *Yt_ev,
                        const float *wv, const int *fit, int nf, const int *ev,
                        const int *mode0, int nev, int ntok,
                        const ds4_loss_weights *lw, double dscale, uint64_t seed,
                        int nth, FILE *lf, float la0, float lc0, float tot0,
                        float *Yhat, float *Cb, float *Cp) {
    uint8_t *isfit = xmalloc((size_t)ntok);
    memset(isfit, 0, (size_t)ntok);
    for (int i = 0; i < nf; i++) isfit[fit[i]] = 1;
    double delta[256];
    float dzf[256];
    solve_ge(zp, R, isfit, ntok, 1e-3, L, delta);
    float *gecorr = ge_corr_build(zp, delta, ntok);
    for (int e = 0; e < 256; e++) dzf[e] = (float)delta[e];
    best_t best = { tot0, 0, 0, 0, 0, la0, lc0, NULL };
    run_bc_arm("GE", dzf, 256, gecorr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
               lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp);
    int inject = best.arm && best.tot < tot0;
    if (inject) {
        float g[256];
        for (int e = 0; e < 256; e++) g[e] = 1.0f + dzf[e];
        emitge_inject(ldir, L, g);
    }
    fprintf(lf, "# emit-ge: %s total=%.6f 裸=%.6f ER=%.2f%%\n",
            inject ? "注入" : "输裸不注入", best.tot, tot0, best.er * 100);
    printf("★L%d GE-only: total %.4f vs 裸 %.4f | ER %+.2f%% → %s\n",
           L, best.tot, tot0, best.er * 100, inject ? "注入 bf.GE" : "输裸, 不注入");
    free(isfit);
    free(gecorr);
}
