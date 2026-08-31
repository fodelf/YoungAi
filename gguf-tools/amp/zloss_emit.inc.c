/* zloss_emit.inc.c — 冠军序列化与注入(只被 zloss_solve.c include)。
 *
 * 体积铁律(用户 2026-08-26 拷问后定): z 与四损失口径的解必须落成有体积文件——
 * 反修产线注入 dql(冻结判决尺原生认) → dql_to_zchain → 引擎 --zchain 加载,
 * 三段同一条链; "只出诊断表"的跑法非法。
 *
 * 两条发射路:
 *   --emit-ge DIR  GE-only 快路(斜率标定/GE 战役): 只解 GE+GEw, 四损失 held 择优,
 *                  赢裸即注入 bf.GE(g=1+δ)。
 *   --emit-z  DIR  全网格路: 四损失选出的冠军臂整体落地——图臂→zl.RRR(±bf.GE 基),
 *                  门臂→bf.GE; 无引擎类型的臂(mul/GEc/多模式)响亮报不落地。
 * 记录=zlayer make_rec 同构(116B 头: 名@0/psz u64@88/vd=1@112); 判决尺语义:
 * 专家累加乘 g_e(ds4quant_run_p4), 学生基线 Ys=Σw·y ⇒ g_e=1+δ_e。
 * 账本 zinject_manifest.txt 同 zlayer(L osz n0, 回滚=按账截断);
 * 四损失入档 fourloss_manifest.txt(每层: 臂/λ/k/四损失前后/ER/权重)。
 */

#include <sys/stat.h>
#include "src/common/ds4_float.h"

/* 多记录注入: recs=连续记录字节, nrec=条数; 幂等闸=账本已有本层即停车 */
static void inject_recs(const char *ldir, int L, const uint8_t *recs, size_t len, int nrec) {
    char dql[1200], man[1200];
    snprintf(dql, sizeof dql, "%s/dql_L%02d.bin", ldir, L);
    snprintf(man, sizeof man, "%s/zinject_manifest.txt", ldir);
    FILE *mf = fopen(man, "r");
    if (mf) {
        char line[256];
        while (fgets(line, sizeof line, mf))
            if (atoi(line) == L && strchr(line, ' '))
                die("L%d 已在注入账本 %s — 工作区不是干净态, 停车", L, man);
        fclose(mf);
    }
    struct stat ds;
    if (stat(dql, &ds)) die("dql 不存在: %s", dql);
    long long osz = (long long)ds.st_size;
    FILE *df = fopen(dql, "r+b");
    if (!df) die("dql 打不开(r+b): %s", dql);
    uint32_t n0;
    if (fseeko(df, 8, SEEK_SET) || fread(&n0, 4, 1, df) != 1) die("dql nrec 读不到");
    if (fseeko(df, 0, SEEK_END) || fwrite(recs, 1, len, df) != len) die("dql 追加失败");
    uint32_t n1 = n0 + (uint32_t)nrec;
    if (fseeko(df, 8, SEEK_SET) || fwrite(&n1, 4, 1, df) != 1) die("dql nrec 回写失败");
    fclose(df);
    mf = fopen(man, "a");
    if (!mf) die("账本写不开: %s", man);
    fprintf(mf, "%d %lld %u\n", L, osz, n0);
    fclose(mf);
}

/* 116B 记录头+载荷(zlayer make_rec 同构), 追加进动态缓冲 */
static uint8_t *rec_append(uint8_t *buf, size_t *len, const char *nm,
                           const void *pay, size_t psz) {
    buf = realloc(buf, *len + DS4_AMP_REC_HDR + psz);
    if (!buf) die("rec OOM");
    uint8_t *r = buf + *len;
    memset(r, 0, DS4_AMP_REC_HDR);
    memcpy(r, nm, strlen(nm));
    uint64_t p64 = psz; memcpy(r + DS4_AMP_REC_OFF_PSZ, &p64, 8);
    int32_t vd = 1; memcpy(r + DS4_AMP_REC_OFF_VD, &vd, 4);
    memcpy(r + DS4_AMP_REC_HDR, pay, psz);
    *len += DS4_AMP_REC_HDR + psz;
    return buf;
}

static uint8_t *rec_ge(uint8_t *buf, size_t *len, const float *dz256) {
    uint16_t g16[256];
    for (int e = 0; e < 256; e++) g16[e] = ds4_f64_to_f16(1.0 + (double)dz256[e]);
    return rec_append(buf, len, "bf.GE", g16, sizeof g16);
}

/* zl.RRR 载荷: k(u32) tr=0.5(f32) din(u32) dout(u32) + f16 z[k]|U[dout×k]|V[din×k]
 * (zlayer_p7 非 ADDON 路同构; tr 槽写 0.5=引擎信任域契约) */
static uint8_t *rec_rrr(uint8_t *buf, size_t *len, const ds4_z *zl) {
    uint32_t K = zl->k, din = zl->d_in, dout = zl->d_out;
    size_t nh = (size_t)K + (size_t)dout * K + (size_t)din * K;
    size_t psz = 16 + nh * 2;
    uint8_t *pay = xmalloc(psz);
    float tr05 = 0.5f;
    memcpy(pay, &K, 4); memcpy(pay + 4, &tr05, 4);
    memcpy(pay + 8, &din, 4); memcpy(pay + 12, &dout, 4);
    uint16_t *h = (uint16_t *)(pay + 16);
    for (uint32_t c = 0; c < K; c++) h[c] = ds4_f64_to_f16((double)zl->z[c]);
    uint16_t *U16 = h + K, *V16 = h + K + (size_t)dout * K;
    for (size_t i = 0; i < (size_t)dout * K; i++) U16[i] = ds4_f64_to_f16((double)zl->U[i]);
    for (size_t i = 0; i < (size_t)din * K; i++) V16[i] = ds4_f64_to_f16((double)zl->V[i]);
    buf = rec_append(buf, len, "zl.RRR", pay, psz);
    free(pay);
    return buf;
}


/* zl.4L 载荷(四损失参数文件, 用户架构: 放大器产/引擎耗, 前向调制+在线调 z 双用):
 * f32 w_align,w_classify,w_smooth,w_fixed | u64 dither_seed | f32 dither_scale |
 * u32 D | f16 wcls[D](fit 侧教师 per-dim 方差=classify 重要性) — 共 8224B。 */
static uint8_t *rec_4l(uint8_t *buf, size_t *len, const ds4_loss_weights *lw,
                       uint64_t seed, float dscale, const float *wclsf, uint32_t d) {
    size_t psz = 16 + 8 + 4 + 4 + (size_t)d * 2;
    uint8_t *pay = xmalloc(psz);
    memcpy(pay, &lw->w_align, 4); memcpy(pay + 4, &lw->w_classify, 4);
    memcpy(pay + 8, &lw->w_smooth, 4); memcpy(pay + 12, &lw->w_fixed, 4);
    memcpy(pay + 16, &seed, 8);
    memcpy(pay + 24, &dscale, 4);
    memcpy(pay + 28, &d, 4);
    uint16_t *h = (uint16_t *)(pay + 32);
    for (uint32_t j = 0; j < d; j++) h[j] = ds4_f64_to_f16((double)wclsf[j]);
    buf = rec_append(buf, len, "zl.4L", pay, psz);
    free(pay);
    return buf;
}

/* 四损失入档: 每层一行(臂/λ/k/裸→冠军四项/ER/损失权重) */
static void fourloss_archive(const char *ldir, int L, const best_t *b, float la0,
                             float lc0, float tot0, const ds4_loss_weights *lw,
                             const char *action) {
    char p[1200]; snprintf(p, sizeof p, "%s/fourloss_manifest.txt", ldir);
    FILE *f = fopen(p, "a"); if (!f) die("四损失档写不开: %s", p);
    fprintf(f, "L%02d arm=%s lam=%.3g k=%d align %.6f->%.6f cls %.6f->%.6f "
            "total %.6f->%.6f ER=%.2f%% w=[%g %g %g %g] %s\n",
            L, b->arm ? b->arm : "-", b->lam, b->k, la0, b->la, lc0, b->lc,
            tot0, b->tot, b->er * 100, lw->w_align, lw->w_classify, lw->w_smooth,
            lw->w_fixed, action);
    fclose(f);
}

/* GE-only 快路一层: GE 与 GEw(cls 白化目标) 双解, 四损失 held 择优, 赢裸注入。 */
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
    best_t best = { tot0, 0, 0, 0, 0, la0, lc0, NULL };
    int tz = g_track_z; g_track_z = 1;   /* 门参数跟踪走 best.bcdz 统一路 */
    double delta[256]; float dzf[256];
    solve_ge(zp, R, isfit, ntok, 1e-3, L, delta, NULL);
    float *gecorr = ge_corr_build(zp, delta, ntok);
    for (int e = 0; e < 256; e++) dzf[e] = (float)delta[e];
    run_bc_arm("GE", dzf, 256, gecorr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
               lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp);
    {
        float *sw2 = mk_sw2_fit(R, Ys, fit, nf);
        double dw[256]; float dwf[256];
        solve_ge(zp, R, isfit, ntok, 1e-3, L, dw, sw2);
        float *gwcorr = ge_corr_build(zp, dw, ntok);
        for (int e = 0; e < 256; e++) dwf[e] = (float)dw[e];
        run_bc_arm("GEw", dwf, 256, gwcorr, X, Ys, Yt_ev, wv, R, ev, mode0, nev,
                   lw, dscale, seed, nth, L, lf, &best, Yhat, Cb, Cp);
        free(gwcorr); free(sw2);
    }
    g_track_z = tz;
    int inject = best.arm && best.bckind == 1 && best.tot < tot0;
    if (inject) {
        uint8_t *recs = NULL; size_t len = 0;
        recs = rec_ge(recs, &len, best.bcdz);
        inject_recs(ldir, L, recs, len, 1);
        free(recs);
    }
    fourloss_archive(ldir, L, &best, la0, lc0, tot0, lw, inject ? "注入bf.GE" : "输裸不注入");
    printf("★L%d GE-only: 冠军 %s total %.4f vs 裸 %.4f | ER %+.2f%% → %s\n",
           L, best.arm ? best.arm : "-", best.tot, tot0, best.er * 100,
           inject ? "注入 bf.GE" : "输裸, 不注入");
    free(isfit);
    free(gecorr);
}

/* 全网格路终局: 四损失冠军整体落地。ge_dz=基 GE δ(组合臂的 bf.GE 基)。 */
static void emit_z_finish(const char *ldir, int L, best_t *best, const float *ge_dz,
                          float la0, float lc0, float tot0, const ds4_loss_weights *lw,
                          const float *Yt, const int *fit, int nf,
                          uint64_t seed, float dscale, int ge_wins) {
    float *wclsf = xmalloc(D * sizeof(float));   /* fit 侧教师 per-dim 方差(引擎 classify 权) */
    {
        float *Ytf = xmalloc((size_t)nf * D * sizeof(float));
        for (int i = 0; i < nf; i++)
            memcpy(Ytf + (size_t)i * D, Yt + (size_t)fit[i] * D, D * sizeof(float));
        ds4_loss_dim_variance(Ytf, (uint32_t)nf, D, wclsf);
        free(Ytf);
    }
    const char *action;
    /* ★GE 与 z 的正确关系(2026-08-27 修正)★ 2026-08-26 曾用 ge_solo 硬凑: 纯 z 臂
     * 夺冠时也补落 GE —— 但纯 z 臂的 z 解的是【完整残差 R】, 再叠 GE 就是同一部分修两遍,
     * 重尾被踩(PPL 比 1.385 反超裸 1.361)。正解=给每个 z 臂都提供 GE 基版本(GE+ftA/
     * ftAw/GE+4L), 让四损失在同口径下择优; 冠军带 GE 基才落 GE, 不再硬凑。 */
    const int ge_solo = 0; (void)ge_wins;
    if (!best->arm || best->tot >= tot0) action = "输裸不注入";
    else if (best->zkeep) {
        uint8_t *recs = NULL; size_t len = 0; int nrec = 0;
        const int with_ge = (best->zkeep_hasge || ge_solo) && ge_dz;
        if (with_ge) { recs = rec_ge(recs, &len, ge_dz); nrec++; }
        recs = rec_rrr(recs, &len, best->zkeep); nrec++;
        recs = rec_4l(recs, &len, lw, seed, dscale, wclsf, D); nrec++;   /* 四损失参数随 z 落地 */
        inject_recs(ldir, L, recs, len, nrec);
        free(recs);
        action = with_ge ? (best->zkeep_hasge ? "注入bf.GE+zl.RRR" : "注入bf.GE(解耦)+zl.RRR")
                         : "注入zl.RRR";
        printf("★L%d emit-z: %s k=%u din=%u → %s (%.2fMB)\n", L, best->arm,
               best->zkeep->k, best->zkeep->d_in, action, len / 1048576.0);
    } else if (best->bckind == 1) {
        uint8_t *recs = NULL; size_t len = 0;
        recs = rec_ge(recs, &len, best->bcdz);
        recs = rec_4l(recs, &len, lw, seed, dscale, wclsf, D);
        inject_recs(ldir, L, recs, len, 2);
        free(recs);
        action = "注入bf.GE";
        printf("★L%d emit-z: 冠军 %s → bf.GE 注入\n", L, best->arm);
    } else {
        action = "★冠军臂无引擎序列化路, 未落地★";
        printf("★L%d emit-z: 冠军 %s (M=%d) 无引擎类型 — 响亮不落地, 挂账\n",
               L, best->arm, best->M);
    }
    fourloss_archive(ldir, L, best, la0, lc0, tot0, lw, action);
    free(wclsf);
    if (best->zkeep) { ds4_z_free(best->zkeep); best->zkeep = NULL; }
}
