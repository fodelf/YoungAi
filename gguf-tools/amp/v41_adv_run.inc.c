/* v41_adv_run.inc.c — 后训练 ③ 第七版(2026-09-29, back.md §14.4): 优势加权的自 token 靶。
 * v41_amp_run.c 单 TU include, 排在 v41_margin_solve.inc.c 之后(复用它的 mg_acc / mg_read_gr / sft_write_base)。
 *
 * 【靶】同一真实请求按模型卡采样 N 份, 次日真实行情打分, 组内减均值得优势 A_i(夜间脚本算好写进清单)。
 *   每份样本【生成段】的每个位置 r 一行: 方向 c_r = ∂log p(y_r)/∂y_L39(出口线性化, v41_klt_own_dirs),
 *   右端 η·A_i, 权重 1。最小二乘 Σ_r (c_r·Δ − η·A_i)² + 岭 ⇒ (Σ c cᵀ + 岭)Δ = η Σ A·c
 *   = Fisher 预条件的策略梯度一步(§13.3): 赢的那种写法每个 token 抬, 输的压; A=0 的行只出现在左端 = 天然的"别处别动"。
 * 【行筛】p(y_r) ≥ own_pmax 的位置 c_r≈0(自 token 就是期望), 不进料 —— 套话位置省掉 ye 缓冲(一行 123 KB)。
 * 【行预算】(--rows-cap, 09-29 内存账) 一份 CFO 样本生成段 2.1~3.4 万行, 4 份 8.5 万行 × 123 KB = 10.5 GB, 而 113.6 GB 的模型驻留后
 *   整机只剩 ~10 GB(core_model_map.c 内存地板) —— 装不下是物理墙。所以每份筛完 p 后按【系统抽样】均匀留 ≤ rows_cap/N 行(等距步长,
 *   确定性; 均匀子集对 Σ A·c 与 Fisher 都是无偏估计, 只是方差大一点), 直接写进设备 ye 缓冲(容量 = rows_cap 行, 不留主机副本)。
 *   12k 行 = 1.5 GB 设备 + 0.25 GB 方向表。
 * 【两遍取料】(09-30 实撞: "只存生成段 bf16 窗"一份 1.7~2.1 GB, 叠上打分状态与解码层懒分配 30 s 把 9 GB 吃到 2.2 GB 被杀)
 *   第一遍不挂钩子, 只出榜单/rms 表(几十 MB 落盘) ⇒ 筛 p(y)、等距抽样定行集 ⇒ 第二遍挂钩子只把这 ≤ rows_cap/N 行搬进 f32 槽
 *   (4000 行 = 0.49 GB), 槽满后逐行进设备 ye。多花一遍前向(分块 512 下一份约 3 分钟), 换回整段窗的 2 GB。取料分块用 --score-chunk
 *   (分块不改输出, 见 v41_amp_run.c 的 score_chunk 注释)。
 * 【清单】一行一份样本: <ids.txt(提示+生成)> <top.bin> <rms.bin> <A_i> <nprompt> <折号>
 *   行 r 预测 ids[r+1]; 生成段 = r ∈ [nprompt−1, n−2]; 部署同路切分点 P = nprompt(提示按生成路预填, 生成段跑满解码器)。
 *   折号只管留一预测(J_out): 多请求夜跑按请求分折, 单请求 demo 按样本分折(同一请求的另一条轨迹算"没解过的")。
 * 【产物】cand_adv_e<η>_l<λ>/gr_L39.bin(fp4x32, 与上一版 ③ 相乘后落盘) + base.fnv + predict.txt; candidates.txt。
 *   ★排序键 = 留一(按折)预测的折外目标 J_out = Σ_{折外行} A·Δlog p(y_r)★: 折外为正 = 学到的东西在没解过的轨迹上
 *   也朝奖励方向推; 真判决仍是挂 ③ 重采(夜间脚本 demo/walk 段), 这里全是【预测】。 */

typedef struct { char ids[1024], top[1024], rms[1024], nll[1100]; float adv; int nprompt, grp; } adv_item;

static adv_item *adv_list_read(const char *path, int *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "★清单打不开 %s★\n", path); return NULL; }
    int cap = 32, n = 0;
    adv_item *it = malloc((size_t)cap * sizeof(adv_item));
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n == cap) { cap *= 2; it = realloc(it, (size_t)cap * sizeof(adv_item)); }
        if (sscanf(line, "%1023s %1023s %1023s %f %d %d", it[n].ids, it[n].top, it[n].rms, &it[n].adv, &it[n].nprompt, &it[n].grp) != 6) {
            fprintf(stderr, "★清单第 %d 行不是 6 列(<ids> <topk> <rms> <A_i> <nprompt> <折号>)★\n", n + 1); fclose(f); free(it); return NULL; }
        snprintf(it[n].nll, sizeof it[n].nll, "%.1023s.nll", it[n].top);
        n++;
    }
    fclose(f);
    if (!n) { fprintf(stderr, "★清单是空的★\n"); free(it); return NULL; }
    *n_out = n;
    return it;
}

/* 累积区(只有每行的元数据, ye 直接进设备、xh 不要)按需翻倍: 保留多少行只有取完料、看过榜单才知道。 */
static int mg_acc_grow(mg_acc *a, int need) {
    if (need <= a->cap) return 0;
    int cap = a->cap ? a->cap : 4096;
    while (cap < need) cap *= 2;
    const int nu = a->nu, K = a->K;
#define GROW(p, per) do { void *t_ = realloc(a->p, (size_t)cap * (per)); if (!t_) { fprintf(stderr, "★累积区扩到 %d 行失败(%s)★\n", cap, #p); return -1; } a->p = t_; } while (0)
    GROW(rw, (size_t)nu * 4); GROW(sel, (size_t)nu * 4);
    GROW(alpha, 4); GROW(inv, 4); GROW(ta, 4); GROW(tb, 4); GROW(m0, 4); GROW(want, 4); GROW(wlp, 4);
    GROW(tid, (size_t)K * 4); GROW(tlp, (size_t)K * 4); GROW(smp, 4); GROW(rowid, 4); GROW(repw, 4);
#undef GROW
    a->cap = cap;
    return 0;
}

/* 一份候选: 解(权重 dw) → 与上一版相乘 → fp4 往返(落盘或不落) → 用落地那份预测每行 Δlog p → 回填 dm(主机)。 */
static int adv_solve_one(ctx_t *c, mg_acc *ab, const float *dC, const int *dsrc, const float *db, const float *dw,
                         float lam, float *dm, const float *sprev, const char *dir, float *hdm, float *stat) {
    const int ns = ab->n, nu = ab->nu, D = ab->D, ne = c->n_expert;
    const size_t nsD = (size_t)ne * D;
    float cg[2] = { 0.f, 0.f };
    if (v41_gr_solve_margin_gpu(c->dye, c->drw, c->dsel, dC, dsrc, db, dw, ns, nu, D, ne, lam, NULL, c->dgate, 1,
                                MG_ITER, c->dgr, dm, cg)) return -1;
    if (cudaMemcpy(c->hgr, c->dgr, nsD * 4, cudaMemcpyDeviceToHost)) return -1;
    for (size_t t = 0; t < nsD; t++) c->hgr[t] = (1.0f + c->hgr[t]) * sprev[t];
    if (dir) { if (gr_write_bin_roundtrip(c->hgr, ne, D, dir, c->layer)) return -1; }
    else { uint8_t *pk = gr_fp4_roundtrip(c->hgr, nsD); if (!pk) return -1; free(pk); }
    double sabs = 0.0;
    for (size_t t = 0; t < nsD; t++) { c->hgr[t] = sprev[t] != 0.f ? c->hgr[t] / sprev[t] - 1.0f : 0.f; sabs += fabs((double)c->hgr[t]); }
    if (cudaMemcpy(c->dgrf, c->hgr, nsD * 4, cudaMemcpyHostToDevice)) return -1;
    if (v41_gr_margin_predict(c->dye, c->drw, c->dsel, dC, dsrc, c->dgrf, ns, nu, D, ne, c->dgate, 1, dm)) return -1;
    if (cudaMemcpy(hdm, dm, (size_t)ns * 4, cudaMemcpyDeviceToHost)) return -1;
    stat[0] = cg[0]; stat[1] = cg[1]; stat[2] = (float)(sabs / (double)nsD);
    return 0;
}

/* 一趟部署同路前向: 提示照生成路预填(CED), 生成段跑满解码器。with_hook = 0 只出榜单/rms 表(第一遍); 1 = 挂钩子按 rye_map 取料(第二遍, 表已在盘上)。 */
static int adv_forward(ctx_t *c, ds4_engine *e, const adv_item *it, const int *ids, int nids, int il, int no_engram, int with_hook) {
    c->layer = il; c->got = 0; c->n = nids;
    ds4_engine_v41_set_score_split(it->nprompt);   /* ★部署同路★: 提示照生成路预填, 生成段跑满解码器(09-23 实撞) */
    ds4_engine_v41_set_amp_dir(c->base_amp);
    ds4_engine_v41_set_posttrain_dir(c->base_pt);
    ds4_engine_v41_set_moe_hook(with_hook ? hook : NULL, c);
    ds4_engine_v41_set_moe_hook_layer(il);
    /* 第二遍: 表已在盘上。切分模式要求 --score-no-logits, 而 no-logits 又要求至少一个小出口(core_v41_api.c 两道闸) ⇒ 给一个
     * 一次性的 rms 路径(4 B/行, 用完删); ★不能指向真 rms 表★: 钩子每块取完 L39 就停车, 这些块的行不会写进去, 会把第一遍的整表盖成残表。 */
    char rms2[1200]; snprintf(rms2, sizeof rms2, "%.1100s.pass2", it->rms);
    if (with_hook) ds4_engine_v41_set_score_aux(NULL, NULL, 0, rms2, 1);
    else ds4_engine_v41_set_score_aux(it->nll, it->top, 64, it->rms, 1);
    const int save_reuse = c->reuse_tab;
    c->reuse_tab = with_hook;   /* 第二遍钩子取完 L39 就停车(不算出口), 见 v41_amp_run.c hook 的返回值 */
    const double tf = now_s();
    const int frc = ds4_engine_v41_score_ids(e, ids, nids, "/dev/null", no_engram, c->score_chunk);
    c->reuse_tab = save_reuse;
    if (with_hook) unlink(rms2);
    ds4_engine_v41_set_score_split(0);
    ds4_engine_v41_set_moe_hook(NULL, NULL);
    ds4_engine_v41_set_score_aux(NULL, NULL, 0, NULL, 0);
    c->t_fwd += now_s() - tf;
    return frc;
}

/* 一份样本(两遍): 第一遍出榜单表 → 生成段筛 p(y) < pmax → 系统抽样 ≤ share 行 → 建行→槽映射 → 第二遍钩子只存这些行 → 元数据进 ab、ye 进设备。
 * 返回留下的行数, <0 失败。 */
static int adv_take_sample(ctx_t *c, ds4_engine *e, const adv_item *it, int m, int nit, int ntok_cap, int no_engram, int il,
                           int share, int cap_rows, mg_acc *ab, int *inited, double t_first) {
    int nids = 0;
    int *ids = sft_ids_read(it->ids, &nids);
    if (!ids) return -1;
    if (nids > ntok_cap || it->nprompt < 1 || it->nprompt >= nids - 1) {
        fprintf(stderr, "★第 %d 份: %d 行 / 提示 %d 不合法(取料缓冲 %d 行)★\n", m + 1, nids, it->nprompt, ntok_cap); free(ids); return -1; }
    const int need_rows = nids - (it->nprompt - 1);
    const double t0 = now_s();
    /* 第一遍: 只出表 */
    free(c->rye_map); c->rye_map = NULL; c->rye_slots = 0; free(c->rye); c->rye = NULL;
    int frc = adv_forward(c, e, it, ids, nids, il, no_engram, 0);
    if (frc != 0) { fprintf(stderr, "★第 %d 份第一遍(榜单)失败 rc=%d★\n", m + 1, frc); free(ids); return -1; }
    ds4_etgd tk;
    if (ds4_etgd_read(it->top, &tk)) { free(ids); return -1; }
    if (tk.n < nids) { fprintf(stderr, "★第 %d 份 top-K 表 %d 行 < ids %d 行★\n", m + 1, tk.n, nids); ds4_etgd_free(&tk); free(ids); return -1; }
    float *inv = sft_rms_read(it->rms, nids);
    if (!inv) { ds4_etgd_free(&tk); free(ids); return -1; }
    /* 候选行: 生成段且 p(y) < pmax; 系统抽样(等距取中点, take==ncand 时逐行)定行集 */
    int *cand = malloc((size_t)need_rows * 4);
    c->rye_map = malloc((size_t)nids * sizeof(int));
    if (!cand || !c->rye_map) { free(cand); ds4_etgd_free(&tk); free(inv); free(ids); return -1; }
    for (int r = 0; r < nids; r++) c->rye_map[r] = -1;
    int ncand = 0, gen = 0;
    double lp_sum = 0.0;
    for (int r = it->nprompt - 1; r < nids - 1; r++) {
        gen++;
        const int t = tk.tgt[r]; const float pt = tk.tgt_p[r];
        if (t < 0 || !(pt > 0.f)) continue;
        lp_sum += log((double)pt);
        if (pt >= c->own_pmax) continue;   /* 自 token 就是期望: 方向≈0, 不占料 */
        cand[ncand++] = r;
    }
    int take = (share > 0 && ncand > share) ? share : ncand;
    if (ab->n + take > cap_rows) { fprintf(stderr, "  ★设备 ye 缓冲只剩 %d 行, 这份只能留 %d★\n", cap_rows - ab->n, cap_rows - ab->n); take = cap_rows - ab->n; }
    for (int j = 0; j < take; j++) c->rye_map[cand[(int)(((long)j * 2 + 1) * ncand / (2L * take))]] = j;
    c->rye_slots = take;
    const double t1 = now_s();
    /* 第二遍: 钩子只存有槽的行 */
    if (take > 0) {
        frc = adv_forward(c, e, it, ids, nids, il, no_engram, 1);
        if (frc != 0 || c->got < need_rows || !c->rye) { fprintf(stderr, "★第 %d 份第二遍(取料)失败(rc=%d, %d 行 < 需要 %d, 槽 %p)★\n", m + 1, frc, c->got, need_rows, (void *)c->rye); free(cand); ds4_etgd_free(&tk); free(inv); free(ids); return -1; }
    }
    free(ids);
    const int nu = c->rye_nu, D = c->D;
    if (!*inited && take > 0) {   /* n_used 只有钩子到过才知道 ⇒ 设备 ye 缓冲在第一份取完后分配; 分不出来就在这里停, 别再取下一份 */
        ab->nu = nu; ab->D = D; ab->K = tk.K; *inited = 1;
        const size_t bytes = (size_t)cap_rows * nu * D * 4;
        if (cudaMalloc((void **)&c->dye, bytes)) { fprintf(stderr, "★设备 ye 缓冲 %d 行 × %d × %d = %.2f GB 分不出来: 调小 --rows-cap★\n", cap_rows, nu, D, (double)bytes / 1e9); free(cand); ds4_etgd_free(&tk); free(inv); return -1; }
        printf("[后训练·第七版] 设备 ye 缓冲 %d 行 × %d × %d = %.2f GB 已分(第一份取料 %.0fs)\n", cap_rows, nu, D, (double)bytes / 1e9, now_s() - t_first);
    }
    int kept = 0;
    for (int j = 0; j < take; j++) {
        const int r = cand[(int)(((long)j * 2 + 1) * ncand / (2L * take))];
        if (mg_acc_grow(ab, ab->n + 1)) { free(cand); ds4_etgd_free(&tk); free(inv); return -1; }
        const int s = ab->n++;
        const int t = tk.tgt[r]; const float pt = tk.tgt_p[r];
        if (cudaMemcpy(c->dye + (size_t)s * nu * D, c->rye + (size_t)j * nu * D, (size_t)nu * D * 4, cudaMemcpyHostToDevice)) { free(cand); ds4_etgd_free(&tk); free(inv); return -1; }
        memcpy(ab->rw + (size_t)s * nu, c->rrw + (size_t)r * nu, (size_t)nu * 4);
        memcpy(ab->sel + (size_t)s * nu, c->rsel + (size_t)r * nu, (size_t)nu * 4);
        ab->alpha[s] = c->ralpha[r]; ab->inv[s] = inv[r];
        ab->ta[s] = t; ab->tb[s] = -1; ab->m0[s] = logf(pt); ab->want[s] = t; ab->wlp[s] = logf(pt);
        ab->smp[s] = m; ab->rowid[s] = r; ab->repw[s] = 1.f;
        for (int z = 0; z < ab->K; z++) {
            const float pv = tk.ps[(size_t)r * tk.K + z];
            ab->tid[(size_t)s * ab->K + z] = (pv > 0.f) ? tk.ids[(size_t)r * tk.K + z] : -1;
            ab->tlp[(size_t)s * ab->K + z] = (pv > 0.f) ? logf(pv) : -1e30f;
        }
        kept++;
    }
    printf("  [样本 %d/%d 折 %d A=%+.4f] 生成段 %d 行, p(y)<%.2f 的 %d 行, 留 %d 行(步长 %.1f), 平均 log p(y) %.4f (榜单 %.0fs + 取料 %.0fs)\n",
           m + 1, nit, it->grp, it->adv, gen, c->own_pmax, ncand, kept, kept ? (double)ncand / kept : 0.0, gen ? lp_sum / gen : 0.0, t1 - t0, now_s() - t1);
    fflush(stdout);
    free(cand); ds4_etgd_free(&tk); free(inv);
    free(c->rye); c->rye = NULL; free(c->rye_map); c->rye_map = NULL; c->rye_slots = 0;   /* 槽一份一放 */
    return kept;
}

static int adv_run(ctx_t *c, ds4_engine *e, const char *list_path, const char *out, int il, int ntok_cap, int no_engram) {
    int nit = 0;
    adv_item *it = adv_list_read(list_path, &nit);
    if (!it) return -1;
    int ngrp = 0;
    long gen_total = 0;
    for (int m = 0; m < nit; m++) {   /* 预扫行数: 设备 ye 容量 = min(封顶, 全部生成段行数) */
        if (it[m].grp + 1 > ngrp) ngrp = it[m].grp + 1;
        int nids = 0; int *ids = sft_ids_read(it[m].ids, &nids); if (!ids) { free(it); return -1; }
        free(ids);
        if (nids > it[m].nprompt) gen_total += nids - (it[m].nprompt - 1);
    }
    const int share = c->rows_cap > 0 ? (c->rows_cap / nit > 0 ? c->rows_cap / nit : 1) : 0;
    const int cap_rows = (int)(share > 0 && (long)share * nit < gen_total ? (long)share * nit : gen_total);
    printf("[后训练·第七版] %d 份样本 / %d 折, 层 L%02d, 行筛 p(y) < %.2f, 行预算 %s(每份 ≤ %d, 生成段共 %ld 行), 网格 η=%s λ=%s\n",
           nit, ngrp, il, c->own_pmax, c->rows_cap > 0 ? "封顶" : "不封", share > 0 ? share : (int)gen_total, gen_total, c->eta_s, c->lam_s);
    mg_acc ab; memset(&ab, 0, sizeof ab);
    int rc = 0, inited = 0;
    double sum_abs_a = 0.0;
    const double t_first = now_s();
    for (int m = 0; m < nit && rc == 0; m++) {
        if (adv_take_sample(c, e, &it[m], m, nit, ntok_cap, no_engram, il, share, cap_rows, &ab, &inited, t_first) < 0) rc = -1;
        else sum_abs_a += fabs((double)it[m].adv);
    }
    free(c->rye_map); c->rye_map = NULL; c->rye_slots = 0; free(c->rye); c->rye = NULL;
    if (rc == 0 && (!inited || ab.n < 1)) { fprintf(stderr, "★没有可用的行★\n"); rc = -1; }
    if (rc == 0 && sum_abs_a <= 0.0) { fprintf(stderr, "★全部样本优势为 0(组内同分): 无可重分, 不解★\n"); rc = -1; }
    if (rc) { if (inited) mg_acc_free(&ab); free(it); return -1; }
    const int ns = ab.n, nu = ab.nu, D = ab.D, K = ab.K, ne = c->n_expert;
    const size_t nsD = (size_t)ne * D;
    printf("[后训练·第七版] 累积 %d 行(设备 ye %.2f GB 已在位); 方向表 %.2f GB\n", ns, (double)ns * nu * D * 4 / 1e9, (double)ns * D * 4 / 1e9);
    /* 设备侧: 榜 + 方向表 + 方程右端/权重 + 恒等门 + 增益缓冲(ye 取料时已直接写进 c->dye) */
    cudaFree(c->drw); cudaFree(c->dsel); c->drw = NULL; c->dsel = NULL;
    cudaFree(c->dgr); c->dgr = NULL; cudaFree(c->dgate); c->dgate = NULL;
    float *dC = NULL, *dPs = NULL, *db = NULL, *dw = NULL, *dm = NULL; int *dIds = NULL, *dsrc = NULL;
    float *hps = malloc((size_t)ns * K * 4), *hb = malloc((size_t)ns * 4), *hw = malloc((size_t)ns * 4), *hdm = malloc((size_t)ns * 4),
          *hdm_full = malloc((size_t)ns * 4), *hone = malloc((size_t)ns * 4), *sprev = malloc(nsD * 4);
    int *hsrc = malloc((size_t)ns * 4);
    if (!c->hgr || !hps || !hb || !hw || !hdm || !hdm_full || !hone || !sprev || !hsrc) { fprintf(stderr, "★主机缓冲失败★\n"); mg_acc_free(&ab); free(it); return -1; }
    for (int i = 0; i < ns; i++) { hsrc[i] = i; hone[i] = 1.f; for (int z = 0; z < K; z++) hps[(size_t)i * K + z] = ab.tlp[(size_t)i * K + z] > -1e29f ? expf(ab.tlp[(size_t)i * K + z]) : 0.f; }
    if (cudaMalloc((void **)&c->drw, (size_t)ns * nu * 4) || cudaMalloc((void **)&c->dsel, (size_t)ns * nu * 4) ||
        cudaMalloc((void **)&dC, (size_t)ns * D * 4) ||
        cudaMalloc((void **)&dPs, (size_t)ns * K * 4) || cudaMalloc((void **)&dIds, (size_t)ns * K * 4) ||
        cudaMalloc((void **)&db, (size_t)ns * 4) || cudaMalloc((void **)&dw, (size_t)ns * 4) || cudaMalloc((void **)&dm, (size_t)ns * 4) ||
        cudaMalloc((void **)&dsrc, (size_t)ns * 4) || cudaMalloc((void **)&c->dgate, (size_t)ns * 4) ||
        cudaMalloc((void **)&c->dgr, nsD * 4) || (!c->dgrf && cudaMalloc((void **)&c->dgrf, nsD * 4))) {
        fprintf(stderr, "★设备缓冲(方向表 %.2f GB)分配失败★\n", (double)ns * D * 4 / 1e9); return -1; }
    CK(cudaMemcpy(c->drw, ab.rw, (size_t)ns * nu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dsel, ab.sel, (size_t)ns * nu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dIds, ab.tid, (size_t)ns * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dPs, hps, (size_t)ns * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsrc, hsrc, (size_t)ns * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dgate, hone, (size_t)ns * 4, cudaMemcpyHostToDevice));
    if (v41_klt_own_dirs(c->klt, ab.ta, dIds, dPs, ab.alpha, ab.inv, ns, K, D, dC)) return -1;
    for (size_t t = 0; t < nsD; t++) sprev[t] = 1.0f;
    if (mg_read_gr(c->base_pt, il, ne, D, sprev) < 0) return -1;
    /* 归一: |A| 最大的样本记 1, η 就是"赢家每个 token 想抬多少 nat"(量纲固定, 大盘 ±1 与个股 ±0.05 两种奖励同一把 η) */
    float amax = 0.f;
    for (int m = 0; m < nit; m++) if (fabsf(it[m].adv) > amax) amax = fabsf(it[m].adv);
    float etas[MG_MAXCAND], lams[MG_MAXCAND];
    const int nE = mg_grid(c->eta_s, etas, MG_MAXCAND), nL = mg_grid(c->lam_s, lams, MG_MAXCAND);
    if (!nE || !nL || nE * nL > MG_MAXCAND) { fprintf(stderr, "★网格 η=%s λ=%s 不合法★\n", c->eta_s, c->lam_s); return -1; }
    char cp[4300]; snprintf(cp, sizeof cp, "%s/candidates.txt", out);
    FILE *cf = fopen(cp, "w");
    if (!cf) { fprintf(stderr, "★候选表写不了 %s★\n", cp); return -1; }
    fprintf(cf, "# 目录 eta lam ★J_out(留一按折, Σ折外 A·Δlogp)★ J_fit 正A行均Δlogp 负A行均Δlogp |s-1|均值 CG残差 迭代 行数 折数\n"
                "# 全是【预测】(落地 fp4 态); 真判决 = 挂 ③ 重采(demo/walk 段)。排序键 = J_out(折数 <2 时 J_out 恒 0, 只能看 J_fit)。\n"
                "# 行预算 --rows-cap %d(每份 ≤ %d 行系统抽样), 行筛 p(y) < %.2f\n", c->rows_cap, share, c->own_pmax);
    int nok = 0;
    for (int a = 0; a < nE; a++) for (int g = 0; g < nL; g++) {
        const float eta = etas[a], lam = lams[g];
        char dir[4300]; snprintf(dir, sizeof dir, "%s/cand_adv_e%g_l%g", out, eta, lam);
        if (mkdir(dir, 0775) && errno != EEXIST) { fprintf(stderr, "★建不了 %s★\n", dir); continue; }
        for (int i = 0; i < ns; i++) { hb[i] = eta * it[ab.smp[i]].adv / amax; hw[i] = 1.f; }
        CK(cudaMemcpy(db, hb, (size_t)ns * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, hw, (size_t)ns * 4, cudaMemcpyHostToDevice));
        float st[4] = { 0 };
        const double t0 = now_s();
        if (adv_solve_one(c, &ab, dC, dsrc, db, dw, lam, dm, sprev, dir, hdm_full, st)) { fprintf(stderr, "★候选 η=%g λ=%g 解算失败★\n", eta, lam); continue; }
        double jfit = 0.0, pos = 0.0, neg = 0.0; int npos = 0, nneg = 0;
        for (int i = 0; i < ns; i++) {
            const float A = it[ab.smp[i]].adv / amax;
            jfit += A * hdm_full[i];
            if (A > 0) { pos += hdm_full[i]; npos++; } else if (A < 0) { neg += hdm_full[i]; nneg++; }
        }
        /* 留一按折: 折外行权重 0(不出方程, 只作预测); 只有 1 折时留一 = 空方程, 跳过 */
        double jout = 0.0;
        for (int gq = 0; gq < ngrp && ngrp >= 2; gq++) {
            int any = 0;
            for (int i = 0; i < ns; i++) { hw[i] = (it[ab.smp[i]].grp == gq) ? 0.f : 1.f; any |= (hw[i] == 0.f); }
            if (!any) continue;
            CK(cudaMemcpy(dw, hw, (size_t)ns * 4, cudaMemcpyHostToDevice));
            float st2[4] = { 0 };
            if (adv_solve_one(c, &ab, dC, dsrc, db, dw, lam, dm, sprev, NULL, hdm, st2)) { fprintf(stderr, "★留一折 %d 失败★\n", gq); break; }
            for (int i = 0; i < ns; i++) if (it[ab.smp[i]].grp == gq) jout += (it[ab.smp[i]].adv / amax) * hdm[i];
        }
        char pq[4300]; snprintf(pq, sizeof pq, "%s/predict.txt", dir);
        FILE *pf = fopen(pq, "w");
        if (pf) {   /* 落地态(全量解, 就是刚落盘那份)的逐行预测, 自检用: 挂上候选重打表比 log p(y) 的真变化 */
            fprintf(pf, "# 样本 折 行号 A 自token logp0 dlogp_pred\n");
            for (int i = 0; i < ns; i++) fprintf(pf, "%d %d %d %.5f %d %.5f %.6f\n", ab.smp[i], it[ab.smp[i]].grp, ab.rowid[i], it[ab.smp[i]].adv, ab.ta[i], ab.wlp[i], hdm_full[i]);
            fclose(pf);
        }
        if (sft_write_base(c, dir)) continue;
        printf("  [η=%g λ=%g] ★J_out %+.4f★ J_fit %+.4f | 正A行均Δlogp %+.5f(%d) 负A行 %+.5f(%d) | |s−1| 均 %.4f | CG 残差 %.2e 迭代 %d (%.1fs)\n",
               eta, lam, jout, jfit, npos ? pos / npos : 0.0, npos, nneg ? neg / nneg : 0.0, nneg, st[2], st[0], (int)st[1], now_s() - t0);
        fprintf(cf, "cand_adv_e%g_l%g %g %g %.6f %.6f %.6f %.6f %.5f %.3e %d %d %d\n", eta, lam, eta, lam, jout, jfit,
                npos ? pos / npos : 0.0, nneg ? neg / nneg : 0.0, st[2], st[0], (int)st[1], ns, ngrp);
        fflush(cf); fflush(stdout); nok++;
    }
    fclose(cf);
    cudaFree(dC); cudaFree(dPs); cudaFree(dIds); cudaFree(db); cudaFree(dw); cudaFree(dm); cudaFree(dsrc);
    free(hps); free(hb); free(hw); free(hdm); free(hdm_full); free(hone); free(sprev); free(hsrc);
    mg_acc_free(&ab); free(it);
    printf("[后训练·第七版] %d 个候选落在 %s; 挑哪个由 demo/walk 段挂上重采定\n", nok, out);
    return nok ? 0 : -1;
}
