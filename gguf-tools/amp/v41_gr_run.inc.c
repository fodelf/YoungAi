/* v41_gr_run.inc.c — 权重侧逐专家反修的驱动侧编排(2026-09-13)。v41_amp_run.c 单 TU include,
 * 拆出来只为守单文件 ≤500 行; 数值全在 v41_gr_solve.cu, 这里只做缓冲、λ 择优、落盘、manifest。
 *
 * 产物 gr_Lnn.bin: 头 <i32 n_expert><i32 D><i32 类型 43=fp4x32>, 接每个专家每个输出通道的
 * 【增益缩放因子】的偏离量 s−1(4.25 bpw, 17 B/32 个数, 1.04 MB/层; f32 是 7.86, f16 是 3.93)。
 * 引擎加载时 +1 还原, 乘进 VQ 载荷里那一行的 g_r。
 * ★解算器直接在 fp4 格点上解, 码字是 GPU 产出的, 这里一个字节都不转换★(2026-09-14, 用户令
 * "反修输出直接就是 fp4, 而不是转换数据")。为什么必须这样而不是"解完再压": 4 bit 的相对精度就是
 * 10% 量级(E2M1 八个幅值格点), 解完再压等于白扔 11% 的修正量且无人接管; 格点直解让每个专家的
 * 量化误差当场进残差, 由后面的专家在同一轮 Gauss-Seidel 里补偿。机理与判据见 fp4.md, 实现见
 * v41_gr_solve.cu 的 gr_fp4_quant_kernel。落盘前拿 src/common 的唯一解码器回读逐位自证。
 * ★存缩放因子而不是新增益★: 盘上原 g 一个字节不动, 插件不挂就是裸底座; 且缩放因子恒在 1 附近,
 * 万一某层解崩了, 看一眼分布就能发现(新增益的绝对值没有这个参照)。 */

/* λ 网格: 岭把解锚在"不改"上, λ 无量纲(尺度已用本通道能量归一)。大 λ ⇒ 解被压回 1。
 * 与低秩路的 SEL_LAMS 不同数量级是应该的 —— 那边是 XᵀX 的正则, 这边是一维带能量归一的岭。
 *
 * ★2026-09-19/20 往下加一格 0.01(用户令"r 正则范围新增支持 0.01、0.1 这两个", 0.1 原网格已有)★
 * 依据不是直觉, 是现役产物自己的账: `gr-fin-40-fp4/manifest.txt` 37 个挂上的层里 ——
 *     λ=0.1(旧下界) 9 层 | λ=1 26 层 | λ=10 2 层 | λ=100 **0 层** | λ=1000 **0 层**
 * 撞下界的那 9 层(L00~L05 + L13 + L19 + L39)恰好是最肥的一批(L00 层内 val +26.7%)。
 * ★选到边界 = 网格在那一侧是错的★: 这些层的最优 λ 很可能在 0.1 以下, 岭把它们的修正量压小了,
 * 而旧网格量不到这件事 —— 它只能报"0.1 最好", 报不了"还想更小"。
 * 网格按用户点名的值走: 只加 0.01, 上界 1000 原样留着(没被选中不等于要砍, 用户没令砍)。
 *
 * ★代价要认★: 每层"N 选 1 取最大"是有选择偏差的(纯噪声层也会挑出一个正的 val), N 从 5 变 6 会让
 * 这个偏差略增, 也就是 GR_GATE(0.5%) 的显著性略降。★不因此动闸★ —— 那个数 09-15 用端到端尺
 * 扫过(见下), 凭直觉调它是拿判决尺的噪声换数字。要判新网格值不值, 看的是端到端主尺, 不是层内 val。
 * ★下一轮的检查点★: 跑完看 manifest 的 λ 分布, 若又有一批层撞新下界 0.01, 再往下扩(0.001)要用户点头。 */
#define GR_NL 6
static const float GR_LAMS[GR_NL] = {0.01f, 0.1f, 1.0f, 10.f, 100.f, 1000.f};
#define GR_ITER 3            /* Gauss-Seidel 轮数: 专家间耦合只来自同 token 的 6 个 pick, 很稀疏 */
/* ★闸(百分点): 层最优 val 不高于它不挂。★这条曲线 2026-09-15 扫完了, 别再调这个旋钮★
 * 金融主尺三个点(Same top / KLD / PPL / 体积):
 *     0(全留 40 层) → 72.08% / 0.57718 / 9.2930 / 41.8 MB   ★四项全负, 唯一明确差的★
 *     0.5(37 层)    → 72.18% / 0.57474 / 9.2545 / 38.6 MB   ← 现役
 *     1.0(35 层)    → 72.08% / 0.57429 / 9.2512 / 36.6 MB
 * 读法: ★0.5 与 1.0 在这把尺上不可区分★ —— Same top 与 p5 说 0.5 赢, KLD 与 PPL 说 1.0 赢,
 * 差距 0.10pp / 0.08% / 0.0033 全在噪声带。曲线不是"0.5 处有峰", 是 0.5~1.0 平坦、0 处掉下去。
 * 也就是: val +0.27~0.47% 那批层有害(0 vs 0.5 的差集), +0.79~0.81% 那批无可测收益也无害。
 * ★为什么停在这里而不是继续找最优★: 同一把判决尺已经选了三次配置, 胜负全在 0.1pp 级 ——
 * 再调就是在选这把尺上的噪声(和"5 个 λ 挑最大"同一个毛病, 只是搬到了更高一层)。保持 0.5 不动。
 * 机理(为什么闸不是省体积而是显著性检验): 每层在 5 个 λ 里挑 val 最大的, "5 选 1 取最大"对无信号层
 * 也会交出一个正数(选择偏差) ⇒ 低于噪声水平的层, 它的 s 是在拟合料的随机涨落上解出来的, 到判决料上是负的。
 * 而序贯让这份伤害传播: 挂上的噪声层改掉下游每层看到的残差, 下游在被污染的基线上解(gate=0 实撞:
 * L31~L37 解全变, L39 层内虚高 5pp 却在链上没兑现)。
 * 【09-15 已用端到端尺验过这个数, 不要再凭直觉动它】用户问"被闸掉的是真没有价值吗", 于是把闸改成 0
 * (大于 0 都留)整趟重跑对照: 40 层全挂的金融主尺 Same top 72.08% / KLD 0.57718 / PPL 9.2930,
 * 而 37 层的是 72.18% / 0.57474 / 9.2545 —— **四项全负, 体积还多 3.1 MB** ⇒ 改回 0.5。
 * 被闸的那批(val +0.27~+0.46%)岭已经把它们压到 |s−1| 均 0.011(肥层是 0.2~0.3), 写进去近乎恒等,
 * 既没帮上忙也没帮倒忙。注意★这不是"三层有害"★: 序贯下被闸的层不落盘, 误差留给后面的层补偿,
 * 所以 gate=0 那趟后面 10 层的解全变了(L39 层内还高 5.0pp 却在链上没兑现), 判的是那个配置整体。
 * 下面这段是当时放开闸的理由, 留着当账: 
 * 原来是 0.5(百分点), 出身是"与低秩路同闸便于横比" —— 不是从端到端推出来的, 而且从没验过:
 * 09-13 那轮直接用了它, 没跑过 40 层 vs 37 层的对照。实撞: 本轮 L32 val +0.46% 被闸掉, 与 0.51%
 * 之间没有任何机理差别, 落在哪边纯粹由这个拍出来的数决定。
 * 为什么现在敢放开: ①体积不再是理由 —— fp4 之后 1.04 MB/层, 三层 +3.1 MB 折底座 +0.003%
 * (f32 时代一层 7.86 MB 还能谈体积); ②岭替我们兜底 —— 真没信号的层最优 λ 会往 1000 走, |s−1| 塌到
 * 0.01 量级, 写进去的修正本来就接近恒等。
 * ★仍然留着 >0 这个下限★: val ≤ 0 = 这一层在 held-out 上被改坏了, 那种层挂上去是负收益, 不是噪声。
 * ★口径提醒★: val 是 held-out 估计量, 每层在 5 个 λ 里挑最大的, 纯噪声层也会挑出一个略大于 0 的数
 * (选择偏差) —— 所以 +0.05% 不等于"确实有肉"。有没有价值由端到端主尺判, 对照产物见 fp4.md §4。 */
#define GR_GATE 0.5f

/* 量化往返: hs(缩放因子, 锚 1) → fp4x32 码字, 并把 hs 原地换成【解回来的那一份】。
 * ★这是后训练(第三件)专用的路★ —— 它解的是"在上一版 ② 之上再乘多少", 落盘的是乘积, 所以只能
 * 先解完再压。权重侧反修(gr 路)★不走这里★: 它的解算器直接在 fp4 格点上解, 码字由 GPU 产出,
 * 见 v41_gr_solve.cu 的"格点直解"。留一泛化尺(尺 L)每折都要往返但不落盘, 所以往返与写盘是两个函数。 */
static uint8_t *gr_fp4_roundtrip(float *hs, size_t nsD) {
    const size_t nblk = nsD / 32u;
    float *dev = malloc(nsD * 4);
    uint8_t *pk = malloc(nblk * 17u);
    if (!dev || !pk) { free(dev); free(pk); fprintf(stderr, "★fp4 缓冲分配失败★\n"); return NULL; }
    for (size_t t = 0; t < nsD; t++) dev[t] = hs[t] - 1.0f;
    ds4_quant_fp4x32(dev, nblk, pk);
    ds4_deq_fp4x32(pk, nblk, dev);
    for (size_t t = 0; t < nsD; t++) hs[t] = 1.0f + dev[t];
    free(dev);
    return pk;
}

/* ★产物落盘: 写的就是【解算器产出的那份码字】, 中间没有任何转换★(2026-09-14)。
 * 头 <i32 n_expert><i32 D><i32 43=fp4x32>, 接 [n_expert][D/32] 个 17 字节块(存的是 s−1)。
 * 写 .part 再改名, 半成品永不被下一遍挂上。
 *
 * ★落盘前必须解码自证★: 码字是 GPU 上编出来的(v41_gr_solve.cu 的 gr_fp4_quant_kernel), 解码用的
 * 是 src/common 的唯一解码器 ds4_deq_fp4x32。两边逐位相等才落盘 —— 不等说明 GPU 编码与全仓格式
 * 基元漂开了(舍入、饱和、块缩放择优任一处), 这种漂移不会让程序报错, 只会让盘上的插件与解算器
 * 以为的那一份不是同一个东西, 判决尺量出来就是一笔无从追查的"莫名其妙掉了"。 */
static int gr_write_bin(const float *hs, const uint8_t *pk, int ne, int D, const char *dir, int il) {
    const size_t nsD = (size_t)ne * D;
    if (nsD % 32u) { fprintf(stderr, "  [L%02d] ★%zu 不是 32 的整数倍★\n", il, nsD); return -1; }
    const size_t nblk = nsD / 32u, nby = nblk * 17u;
    float *back = malloc(nsD * 4);
    if (!back) { fprintf(stderr, "  [L%02d] ★自证缓冲分配失败★\n", il); return -1; }
    ds4_deq_fp4x32(pk, nblk, back);
    for (size_t t = 0; t < nsD; t++) {
        const float want = 1.0f + back[t];
        if (want != hs[t]) {
            fprintf(stderr, "  [L%02d] ★码字解回来与解算态对不上(第 %zu 个: 盘 %.9g vs 解 %.9g), 停车★\n",
                    il, t, want, hs[t]);
            free(back); return -1;
        }
    }
    free(back);
    char p[4300]; snprintf(p, sizeof p, "%s/gr_L%02d.bin.part", dir, il);
    FILE *f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "  [L%02d] ★写不了 %s★\n", il, p); return -1; }
    const int32_t hd[3] = { ne, D, (int32_t)DS4_GGT_FP4X32 };
    const int okw = fwrite(hd, 4, 3, f) == 3 && fwrite(pk, 1, nby, f) == nby;
    if (fclose(f) || !okw) { fprintf(stderr, "  [L%02d] ★gr_L%02d.bin 写盘截断★\n", il, il); remove(p); return -1; }
    char q[4300]; snprintf(q, sizeof q, "%s/gr_L%02d.bin", dir, il);
    if (rename(p, q)) { fprintf(stderr, "  [L%02d] ★改名失败★\n", il); return -1; }
    return 0;
}

/* 后训练(第三件)那条路仍是"解完再压", 它落盘的是与上一版相乘之后的 s ⇒ 先往返再走这个写盘函数。 */
static int gr_write_bin_roundtrip(float *hs, int ne, int D, const char *dir, int il) {
    uint8_t *pk = gr_fp4_roundtrip(hs, (size_t)ne * D);
    if (!pk) return -1;
    const int rc = gr_write_bin(hs, pk, ne, D, dir, il);
    free(pk);
    return rc;
}

/* 解一层的增益: 扫 λ → 只认 val → 落盘。返回 <0 失败, 0 跳过, 1 挂上。
 * ★这里报的 train/val 就是落地态★(2026-09-14 改格点直解之后): 解算器在 fp4 格点上解, 最终残差
 * 是拿格点值算的。以前是"按解算态 val 选 λ, 再把赢家压成 fp4", 两件事口径不一样 —— 大 λ 的解
 * 幅度小、相对量化误差更大, 两种口径可能选出不同的 λ, 而部署里跑的永远是压过之后的那一份。 */
static int solve_layer_gr(ctx_t *c, int il, double t0, double t1) {
    /* 行数用 nsel(--fit-rows/--val-rows 选的子集); 没选子集时就是全部取料行。 */
    const int n = c->nsel > 0 ? c->nsel : c->n, D = c->D, nu = c->n_used, ne = c->n_expert;
    const size_t nsD = (size_t)ne * D, nby = nsD / 32u * 17u;
    float best_tr = 0.f, best_va = -1e9f, best_lam = 0.f;
    for (int l = 0; l < GR_NL; l++) {
        float tr = 0.f, va = 0.f;
        if (v41_gr_solve_layer_gpu(c->dye, c->drw, c->dsel, c->dYfp, c->dysh,
                                   n, c->nfit, nu, D, ne, GR_LAMS[l], GR_ITER, c->dgr, c->dgrpk, &tr, &va)) {
            fprintf(stderr, "  [L%02d] ★增益解算失败(λ=%.3g), 停车★\n", il, GR_LAMS[l]);
            return -1;
        }
        printf("  [L%02d] λ=%-5.3g train %+6.2f%%  val %+6.2f%% (fp4 格点上的落地读数)\n", il, GR_LAMS[l], tr, va);
        fflush(stdout);
        /* 赢家的 s 与【它那一份码字】一起留下来 —— 两者必须是同一次解算的产物, 否则落盘自证会炸 */
        if (va > best_va) { best_va = va; best_tr = tr; best_lam = GR_LAMS[l];
                            CK(cudaMemcpy(c->hgr, c->dgr, nsD * 4, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(c->hgrpk, c->dgrpk, nby, cudaMemcpyDeviceToHost)); }
    }
    const double t2 = now_s(); c->t_solve += t2 - t1;
    {   /* 不挂也要报缩放因子的范围 —— val 崩的时候第一件要看的就是解有没有跑飞 */
        double mn = 1e30, mx = -1e30;
        for (size_t t = 0; t < nsD; t++) { const double v = c->hgr[t]; if (v < mn) mn = v; if (v > mx) mx = v; }
        printf("  [L%02d] 最优 λ=%.3g 的缩放因子范围 [%.4f, %.4f]\n", il, best_lam, mn, mx);
    }
    if (best_va <= GR_GATE) {   /* ★<= 不是 <★: 正好等于闸的层没有留的理由 */
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip gr - - %+.2f %+.2f -\n", il, best_tr, best_va); fflush(c->mf);
        printf("  [L%02d] val 最优 %+.2f%% ≤ 闸 %.1f%%, 不挂  (fp %.1fs 解 %.1fs)\n", il, best_va, GR_GATE, t1 - t0, t2 - t1);
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
    if (gr_write_bin(c->hgr, c->hgrpk, ne, D, c->out_dir, il)) return -1;

    /* ★必须记进 ok_layers★: 多遍序贯靠它决定下一遍挂不挂本目录(v41_amp_run.c 的 set_amp_dir)。
     * 漏了这一笔 = 每遍都在裸底座上解, 层与层之间的耦合完全没闭合 —— 而且不会报错, 只是解出来的
     * 东西不是序贯的(09-13 实撞: 三层读数全是独立解的假序贯)。 */
    c->ok_layers++; c->val_sum += best_va; c->dz_sum += sabs / (double)nsD;
    fprintf(c->mf, "L%02d gr %.3g %.2f %+.2f %+.2f [%.4f,%.4f]\n", il, best_lam, sabs / (double)nsD, best_tr, best_va, smin, smax);
    fflush(c->mf);
    printf("  [L%02d] ★挂上★ λ=%.3g train %+.2f%% val %+.2f%% | 缩放 |s−1| 均 %.4f 范围 [%.4f, %.4f] | fp4 %.2f MB  (fp %.1fs 解 %.1fs)\n",
           il, best_lam, best_tr, best_va, sabs / (double)nsD, smin, smax, (double)nsD / 32.0 * 17.0 / 1e6, t1 - t0, t2 - t1);
    v41_st_release_idle(&c->S);
    return 1;
}
