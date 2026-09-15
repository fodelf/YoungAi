/* v41_amp_lowrank.inc.c — 低秩放大器(amp_Lnn.bin: y += x·(B·A))那条路的逐层解算, v41_amp_run.c 单 TU include。
 * 拆出来只为守单文件 ≤500 行。这是 09-12 的第一形态, 09-13 被权重侧逐专家增益(v41_gr_run.inc.c)取代 ——
 * 留着是因为 --target kl 蒸馏靶与 K 扫描仍走这条路, 且它是"层内赢但端到端不赢"那段账的活证据。 */
/* 解第 il 层: 置换 → 上传 → FP 靶 → K×λ 扫描 → 择优 → 解 → 落盘 + manifest。返回 <0 失败, 0 跳过, 1 挂上 */
static int solve_layer(ctx_t *c, int il) {
    const int n = c->n, D = c->D, nu = c->n_used;
    const double t0 = now_s();
    if (!c->klt && resolve_layer(c, il)) return -1;   /* 蒸馏靶不碰 HF 出厂权重: 靶只来自两份 logits */
    if (c->want_ye) {
        /* ★机制审计★ 逐专家输出必须能逐位重建引擎这一层的输出: y[i][d] == Σ_k rw[i][k]·ye[i][k][d]。
         * 对不上就说明取的不是本层的 ys、或 inv 配对错位 —— 那样解出来的 g 全是假账, 且不会报错。
         * 阈值 1e-5: f32 累加序不同(kernel 定序 vs 这里顺序)只该差到 1e-7 量级, 留两个数量级余量。 */
        double num = 0, den = 0; long same = 0, tot = 0;
        for (int i = 0; i < n; i++) for (int d = 0; d < D; d++) {
            float s = c->rysh[(size_t)i * D + d];
            for (int k = 0; k < nu; k++) s += c->rrw[(size_t)i * nu + k] * c->rye[((size_t)i * nu + k) * D + d];
            const float sb = bf16r(s), yv = c->ry[(size_t)i * D + d];
            num += (double)(sb - yv) * (sb - yv); den += (double)yv * yv;
            tot++; if (sb == yv) same++;
        }
        const double rel = den > 0 ? sqrt(num / den) : 0;
        printf("[L%02d] 取料自检: round_bf16(Σ_k rw·ye + ysh) vs y — 逐位同 %.4f%%, 相对差 %.3e %s\n",
               il, 100.0 * same / tot, rel, rel < 1e-4 ? "✓" : "★不符★");
        fflush(stdout);
        if (!(rel < 1e-4)) return -1;
    }
    const int ns = c->nsel > 0 ? c->nsel : n;   /* 解算行数: --fit-rows/--val-rows 选了子集时 < n */
    for (int i = 0; i < ns; i++) {   /* 行置换: 拟合行在前、val 行在后 —— 解算只看前 nfit 行(连续切片) */
        const int r = c->perm[i];
        memcpy(c->hx + (size_t)i * D, c->rx + (size_t)r * D, (size_t)D * 4); memcpy(c->hy + (size_t)i * D, c->ry + (size_t)r * D, (size_t)D * 4);
        memcpy(c->hsel + (size_t)i * nu, c->rsel + (size_t)r * nu, (size_t)nu * 4); memcpy(c->hrw + (size_t)i * nu, c->rrw + (size_t)r * nu, (size_t)nu * 4);
        c->halpha[i] = c->ralpha[r];
    }
    if (c->want_ye) {   /* ye/ysh 也按同一个置换搬 —— 行序配错不会报错, 只会出假账 */
        for (int i = 0; i < ns; i++) {
            const int r = c->perm[i];
            memcpy(c->hye + (size_t)i * nu * D, c->rye + (size_t)r * nu * D, (size_t)nu * D * 4);
            memcpy(c->hysh + (size_t)i * D, c->rysh + (size_t)r * D, (size_t)D * 4);
        }
        CK(cudaMemcpy(c->dye, c->hye, (size_t)ns * nu * D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->dysh, c->hysh, (size_t)ns * D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->drw, c->hrw, (size_t)ns * nu * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->dsel, c->hsel, (size_t)ns * nu * 4, cudaMemcpyHostToDevice));
    }
    const size_t nD = (size_t)ns * D;
    CK(cudaMemcpy(c->dX, c->hx, nD * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dYq, c->hy, nD * 4, cudaMemcpyHostToDevice));
    double qn2 = 0.0; for (size_t i = 0; i < nD; i++) qn2 += (double)c->hy[i] * c->hy[i];
    const double qnorm = sqrt(qn2);
    if (c->klt) {
        float st4[4];
        if (v41_klt_target(c->klt, c->kl_ref, c->kl_stu, c->ralpha, c->perm, ns, D, qnorm, c->eta_rel, c->dYq, c->dYfp, st4)) {
            fprintf(stderr, "  [L%02d] ★蒸馏靶计算失败, 停车★\n", il); return -1;
        }
        printf("  [L%02d] 蒸馏靶: ‖α·g‖/‖y_q‖=%.4f → 按 --eta-rel 缩到 %.4f; α 范围 [%.4f, %.4f] 均值 %.4f\n",
               il, st4[0], c->eta_rel, st4[1], st4[2], st4[3]);
        if (st4[1] <= 0.f) printf("  [L%02d] ★α 跨零: 有行的靶方向是反的, 记账不停车(判决时看端到端)★\n", il);
    } else if (v41_fp_moe_layer(c->ex, c->n_expert, &c->sh, c->dX, c->hsel, c->hrw, ns, D, c->MID, nu, c->clamp, c->dYfp)) {
        fprintf(stderr, "  [L%02d] ★FP 靶计算失败, 停车★\n", il); return -1;
    }
    const double t1 = now_s(); c->t_fp += t1 - t0;
    if (c->want_ye) return solve_layer_gr(c, il, t0, t1);
    float tr[NL * NK], va[NL * NK], trw[NL * NK], vaw[NL * NK], rel = 0.f;
    if (v41_amp_scan_k_gpu(c->dX, c->dYfp, c->dYq, n, D, c->nfit, SEL_KS, NK, SEL_LAMS, NL, tr, va, &rel, c->whiten, trw, vaw)) { fprintf(stderr, "  [L%02d] ★K×λ 扫描失败, 停车★\n", il); return -1; }
    const double t2 = now_s(); c->t_scan += t2 - t1;
    float pct[NL][NK], pct_w[NL][NK], pct_tr[NL][NK];
    for (int l = 0; l < NL; l++) for (int k = 0; k < NK; k++) {
        pct[l][k] = 100.f * (1.f - va[l * NK + k]); pct_tr[l][k] = 100.f * (1.f - tr[l * NK + k]);
        pct_w[l][k] = c->whiten ? 100.f * (1.f - vaw[l * NK + k]) : pct[l][k];
    }
    float best; int ki, lj;
    const int has = choose(c->whiten ? pct_w : pct, &best, &ki, &lj);   /* 白化时按白化 val 择优, 原始 val 照记 */
    if (!has || best < GATE_PCT) {
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip - %.4f - %+.2f -\n", il, rel, best); fflush(c->mf);
        printf("  [L%02d] 靶/‖y_q‖=%.3f | val 最优 %+.2f%% < 闸 %.1f%%, 不挂  (fp %.1fs 扫 %.1fs)\n", il, rel, best, GATE_PCT, t1 - t0, t2 - t1);
        v41_st_release_idle(&c->S);
        return 0;
    }
    const int Kc = SEL_KS[ki]; const float lam = SEL_LAMS[lj];
    float ratio = 1.f;
    const int rc = v41_amp_solve_layer_gpu(c->dX, c->dYfp, c->dYq, c->nfit, D, Kc, lam, c->dA, c->dB, &ratio, c->whiten);
    const double t3 = now_s(); c->t_solve += t3 - t2;
    if (rc != 0) {
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip - %.4f - %+.2f - (解算自检不过 残差比 %.3f)\n", il, rel, best, ratio); fflush(c->mf);
        printf("  [L%02d] ★解算自检不过(残差比 %.3f), 不挂★\n", il, ratio);
        v41_st_release_idle(&c->S);
        return 0;
    }
    const size_t KD = (size_t)Kc * D;
    CK(cudaMemcpy(c->hA, c->dA, KD * 4, cudaMemcpyDeviceToHost));   /* A 行主序 [K][D] */
    CK(cudaMemcpy(c->hB, c->dB, KD * 4, cudaMemcpyDeviceToHost));   /* B 列主序 D×K = 内存 [K][D](与离线格式/引擎加载同) */
    /* ★fp4x32 往返必须在判决之前★(2026-09-13): 落盘格式是 4.25 bpw 的 fp4x32, 部署时引擎解出来的
     * 是【量化过的】A/B。所以这里编码完立刻解回来覆盖 hA/hB 并推回设备 —— 下面的 val 重算与
     * ‖Δ‖ 信任域量的都是部署那一份。少了这一步, 报的是 f32 解的成绩而挂上去的是 fp4 的东西,
     * 又是一笔"内部好看、落地劣化"的账(本仓吃过亏)。 */
    if (KD % 32u) { fprintf(stderr, "  [L%02d] ★K×D=%zu 不是 32 的整数倍, fp4x32 装不下★\n", il, KD); return -1; }
    const size_t nblk = KD / 32u, nby = nblk * 17u;
    ds4_quant_fp4x32(c->hA, nblk, c->pkA); ds4_deq_fp4x32(c->pkA, nblk, c->hA);
    ds4_quant_fp4x32(c->hB, nblk, c->pkB); ds4_deq_fp4x32(c->pkB, nblk, c->hB);
    CK(cudaMemcpy(c->dA, c->hA, KD * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dB, c->hB, KD * 4, cudaMemcpyHostToDevice));
    /* ‖Δ‖/‖y_q‖ 信任域 + fp4 态 val 重算: 在 y_q 副本上应用一次, 再在 val 行(置换后是连续的尾段)
     * 上量 ‖(y_fp−y_q−Δ)‖/‖y_fp−y_q‖ —— 与扫描的 val 同式, 差别只在 A/B 过了 fp4。 */
    float dz = 0.f;
    CK(cudaMemcpy(c->dYtmp, c->dYq, nD * 4, cudaMemcpyDeviceToDevice));
    if (v41_amp_apply_gpu(c->dX, c->dYtmp, c->dA, c->dB, n, D, Kc, &dz)) return -1;
    const int nval = n - c->nfit;
    const size_t voff = (size_t)c->nfit * D, vlen = (size_t)nval * D;
    CK(cudaMemcpy(c->hvf, c->dYfp + voff, vlen * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c->hvt, c->dYtmp + voff, vlen * 4, cudaMemcpyDeviceToHost));
    double rn2 = 0.0;
    for (size_t i = 0; i < vlen; i++) { const double d0 = (double)c->hvf[i] - c->hvt[i]; rn2 += d0 * d0; }
    CK(cudaMemcpy(c->hvt, c->dYq + voff, vlen * 4, cudaMemcpyDeviceToHost));
    double bn2 = 0.0;
    for (size_t i = 0; i < vlen; i++) { const double d0 = (double)c->hvf[i] - c->hvt[i]; bn2 += d0 * d0; }
    const float val4 = bn2 > 0.0 ? (float)(100.0 * (1.0 - sqrt(rn2 / bn2))) : 0.f;
    if (val4 < GATE_PCT) {   /* ★闸判在 fp4 态★: f32 解过闸但量化后不过, 说明这一层的收益全在 fp4 装不下的精度里 */
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip - %.4f - %+.2f - (fp4 后 val %+.2f < 闸)\n", il, rel, best, val4);
        printf("  [L%02d] 选 λ=%g K=%d val %+.1f%% → ★fp4 往返后掉到 %+.2f%% < 闸 %.1f%%, 不挂★\n", il, lam, Kc, pct[lj][ki], val4, GATE_PCT);
        fflush(c->mf); v41_st_release_idle(&c->S);
        return 0;
    }
    char po[4300], pt[4300]; snprintf(po, sizeof po, "%s/amp_L%02d.bin", c->out_dir, il); snprintf(pt, sizeof pt, "%s.part", po);
    FILE *f = fopen(pt, "wb");
    if (!f) { fprintf(stderr, "★写不了 %s★\n", pt); return -1; }
    const int32_t hd[3] = { D, Kc, DS4_GGT_FP4X32 };   /* 头第三字段 = 存储类型, 引擎按它分派(43=fp4x32) */
    fwrite(hd, 4, 3, f); fwrite(c->pkA, 1, nby, f); fwrite(c->pkB, 1, nby, f);
    if (fclose(f) || rename(pt, po)) { fprintf(stderr, "★落盘 %s 失败★\n", po); return -1; }   /* 写完才改名: 半成品永不被下一遍挂上 */
    const double relz = qnorm > 0 ? dz / qnorm : 0.0, tr_pct = 100.0 * (1.0 - ratio);
    c->ok_layers++; c->kcount[ki]++; c->val_sum += val4; c->dz_sum += relz;
    fprintf(c->mf, "L%02d %g %d %.4f %+.2f %+.2f %.4f", il, lam, Kc, rel, tr_pct, val4, relz);
    if (c->whiten) fprintf(c->mf, " %+.2f", pct_w[lj][ki]);
    fprintf(c->mf, "\n"); fflush(c->mf);
    printf("  [L%02d] 靶/‖y_q‖=%.3f | 选 λ=%g K=%d: train %+.1f%% val %+.1f%% → ★fp4 后 %+.1f%%★%s [最优 %+.1f%%]  ‖Δ‖/‖y_q‖=%.4f  (fp %.1fs 扫 %.1fs 解 %.1fs)\n",
           il, rel, lam, Kc, tr_pct, pct[lj][ki], val4, c->whiten ? " (白化择优)" : "", best, relz, t1 - t0, t2 - t1, t3 - t2);
    v41_st_release_idle(&c->S);
    return 1;
}
