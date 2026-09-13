/* v41_gr_run.inc.c — 权重侧逐专家反修的驱动侧编排(2026-09-13)。v41_amp_run.c 单 TU include,
 * 拆出来只为守单文件 ≤500 行; 数值全在 v41_gr_solve.cu, 这里只做缓冲、λ 择优、落盘、manifest。
 *
 * 产物 gr_Lnn.bin: 头 <i32 n_expert><i32 D><i32 类型(1=f32 / 2=f16)>, 接 s[n_expert][D] —— 每个专家每个
 * 输出通道的【增益缩放因子】, 锚 1(=不改)。引擎加载时把它乘进 VQ 载荷里那一行的 g_r。
 * ★落 f16★: 缩放因子恒在 1 附近(实测 |s−1| 中位 0.09~0.31), f16 在 1 附近的分辨率是 2^-10 ≈ 0.001,
 * 相对精度 0.1% —— 比它要修的量化误差(幅值 30~80%)小三个数量级, 无损可言。体积因此从 315 MB(40 层 f32)
 * 降到 157 MB(+0.0023 bpw)。
 * ★存缩放因子而不是新增益★: 盘上原 g 一个字节不动, 插件不挂就是裸底座; 且缩放因子恒在 1 附近,
 * 万一某层解崩了, 看一眼分布就能发现(新增益的绝对值没有这个参照)。 */

/* λ 网格: 岭把解锚在"不改"上, λ 无量纲(尺度已用本通道能量归一)。大 λ ⇒ 解被压回 1。
 * 与低秩路的 SEL_LAMS 不同数量级是应该的 —— 那边是 XᵀX 的正则, 这边是一维带能量归一的岭。 */
#define GR_NL 5
static const float GR_LAMS[GR_NL] = {0.1f, 1.0f, 10.f, 100.f, 1000.f};
#define GR_ITER 3            /* Gauss-Seidel 轮数: 专家间耦合只来自同 token 的 6 个 pick, 很稀疏 */
#define GR_GATE 0.5f         /* 层最优 val 低于此(百分点)不挂 —— 与低秩路同闸, 便于横比 */

/* 解一层的增益: 扫 λ → 只认 val → 落盘。返回 <0 失败, 0 跳过, 1 挂上。 */
static int solve_layer_gr(ctx_t *c, int il, double t0, double t1) {
    const int n = c->n, D = c->D, nu = c->n_used, ne = c->n_expert;
    const size_t nsD = (size_t)ne * D;
    float best_tr = 0.f, best_va = -1e9f, best_lam = 0.f;
    for (int l = 0; l < GR_NL; l++) {
        float tr = 0.f, va = 0.f;
        if (v41_gr_solve_layer_gpu(c->dye, c->drw, c->dsel, c->dYfp, c->dysh,
                                   n, c->nfit, nu, D, ne, GR_LAMS[l], GR_ITER, c->dgr, &tr, &va)) {
            fprintf(stderr, "  [L%02d] ★增益解算失败(λ=%.3g), 停车★\n", il, GR_LAMS[l]);
            return -1;
        }
        printf("  [L%02d] λ=%-5.3g train %+6.2f%%  val %+6.2f%%\n", il, GR_LAMS[l], tr, va);
        fflush(stdout);
        if (va > best_va) { best_va = va; best_tr = tr; best_lam = GR_LAMS[l];
                            CK(cudaMemcpy(c->hgr, c->dgr, nsD * 4, cudaMemcpyDeviceToHost)); }
    }
    const double t2 = now_s(); c->t_solve += t2 - t1;
    {   /* 不挂也要报缩放因子的范围 —— val 崩的时候第一件要看的就是解有没有跑飞 */
        double mn = 1e30, mx = -1e30;
        for (size_t t = 0; t < nsD; t++) { const double v = c->hgr[t]; if (v < mn) mn = v; if (v > mx) mx = v; }
        printf("  [L%02d] 最优 λ=%.3g 的缩放因子范围 [%.4f, %.4f]\n", il, best_lam, mn, mx);
    }
    if (best_va < GR_GATE) {
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip gr - - %+.2f %+.2f -\n", il, best_tr, best_va); fflush(c->mf);
        printf("  [L%02d] val 最优 %+.2f%% < 闸 %.1f%%, 不挂  (fp %.1fs 解 %.1fs)\n", il, best_va, GR_GATE, t1 - t0, t2 - t1);
        v41_st_release_idle(&c->S);
        return 0;
    }
    /* 缩放因子的分布也报出来 —— 全是 1 说明岭把解压死了(那时 val 也该接近 0, 两者对不上就是 bug) */
    double smin = 1e30, smax = -1e30, sabs = 0;
    for (size_t t = 0; t < nsD; t++) {
        const double v = c->hgr[t];
        if (v < smin) smin = v;
        if (v > smax) smax = v;
        sabs += fabs(v - 1.0);
    }
    char p[4300]; snprintf(p, sizeof p, "%s/gr_L%02d.bin.part", c->out_dir, il);
    FILE *f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "  [L%02d] ★写不了 %s★\n", il, p); return -1; }
    const int32_t hd[3] = { ne, D, 2 };   /* 2 = f16 */
    uint16_t *h16 = malloc(nsD * 2);
    if (!h16) { fclose(f); fprintf(stderr, "  [L%02d] ★f16 缓冲分配失败★\n", il); return -1; }
    for (size_t t = 0; t < nsD; t++) h16[t] = ds4_f64_to_f16((double)c->hgr[t]);
    const int okw = fwrite(hd, 4, 3, f) == 3 && fwrite(h16, 2, nsD, f) == nsD;
    free(h16);
    fclose(f);
    if (!okw) { fprintf(stderr, "  [L%02d] ★gr_L%02d.bin 写盘截断★\n", il, il); remove(p); return -1; }
    char q[4300]; snprintf(q, sizeof q, "%s/gr_L%02d.bin", c->out_dir, il);
    if (rename(p, q)) { fprintf(stderr, "  [L%02d] ★改名失败★\n", il); return -1; }

    /* ★必须记进 ok_layers★: 多遍序贯靠它决定下一遍挂不挂本目录(v41_amp_run.c 的 set_amp_dir)。
     * 漏了这一笔 = 每遍都在裸底座上解, 层与层之间的耦合完全没闭合 —— 而且不会报错, 只是解出来的
     * 东西不是序贯的(09-13 实撞: 三层读数全是独立解的假序贯)。 */
    c->ok_layers++; c->val_sum += best_va; c->dz_sum += sabs / (double)nsD;
    fprintf(c->mf, "L%02d gr %.3g %.2f %+.2f %+.2f [%.4f,%.4f]\n", il, best_lam, sabs / (double)nsD, best_tr, best_va, smin, smax);
    fflush(c->mf);
    printf("  [L%02d] ★挂上★ λ=%.3g train %+.2f%% val %+.2f%% | 缩放 |s−1| 均 %.4f 范围 [%.4f, %.4f] | %.1f MB  (fp %.1fs 解 %.1fs)\n",
           il, best_lam, best_tr, best_va, sabs / (double)nsD, smin, smax, (double)nsD * 2 / 1e6, t1 - t0, t2 - t1);
    v41_st_release_idle(&c->S);
    return 1;
}
