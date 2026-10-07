/* core_draft_kd.c — 草稿器蒸馏的主循环: --draft-train <配置>(2026-10-07)。总述见 core_draft_kd.h。
 *
 * 一趟 = 读料(缓存没有就跑底座取) → 第 0 步留出评估(件为零 = 部署态草稿器: ★陪审团首位 Σmin 必须复现 d1 accjury 在同一份 ids 上的数★,
 * 这是接线门, 对不上就停车不开训) → 若干轮: 训练块打乱成批、每批一次 Adam → 每轮末留出评估 + 存 ckpt_eNN → 落件。
 * 判据(全打印, 判读归人): 留出文本块内五位各自的 Σmin(采样下期望接受率)/ p(argmax q)(点质量)/ 贪心一致 —— 首位就是在线投机的 p1,
 * 五位的乘积链就是一轮期望接受数的上界; 训练 KL 只作诊断。 */
#include "core_draft_kd.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

static char *dk_slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)n + 1);
    if (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    return b;
}
static void dk_rstrip(char *s) { size_t n = strlen(s); while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = 0; }

bool dk_cfg_load(dk_cfg *c, const char *path) {
    memset(c, 0, sizeof *c);
    c->rank = 64; c->rank_exit = 64; c->topk = 1024; c->epochs = 3; c->batch_blocks = 128; c->logits_rows = 128; c->seed = 1;
    c->lr = 1e-3f; c->clip = 1.0f; c->init_std = 0.f; c->temp = 1.0f; c->prompt_tail = 0; c->lr_markov = 0.f;
    char *b = dk_slurp(path);
    if (!b) { fprintf(stderr, "ds4: --draft-train 读不了配置 %s\n", path); return false; }
    for (char *ln = strtok(b, "\n"); ln; ln = strtok(NULL, "\n")) {
        while (*ln == ' ' || *ln == '\t') ln++;
        if (!*ln || *ln == '#') continue;
        char *eq = strchr(ln, '=');
        if (!eq) { fprintf(stderr, "ds4: --draft-train 配置行没有 '=': %s\n", ln); free(b); return false; }
        *eq = 0; char *k = ln, *v = eq + 1; dk_rstrip(k); dk_rstrip(v);
        while (*v == ' ') v++;
        if (!strcmp(k, "texts")) snprintf(c->texts, sizeof c->texts, "%s", v);
        else if (!strcmp(k, "out")) snprintf(c->out, sizeof c->out, "%s", v);
        else if (!strcmp(k, "cache")) snprintf(c->cache, sizeof c->cache, "%s", v);
        else if (!strcmp(k, "init")) snprintf(c->init, sizeof c->init, "%s", v);
        else if (!strcmp(k, "rank")) c->rank = (uint32_t)atoi(v);
        else if (!strcmp(k, "rank_exit")) c->rank_exit = (uint32_t)atoi(v);
        else if (!strcmp(k, "topk")) c->topk = (uint32_t)atoi(v);
        else if (!strcmp(k, "epochs")) c->epochs = (uint32_t)atoi(v);
        else if (!strcmp(k, "batch_blocks")) c->batch_blocks = (uint32_t)atoi(v);
        else if (!strcmp(k, "logits_rows")) c->logits_rows = (uint32_t)atoi(v);
        else if (!strcmp(k, "seed")) c->seed = (uint32_t)atoi(v);
        else if (!strcmp(k, "gradcheck")) c->gradcheck = (uint32_t)atoi(v);
        else if (!strcmp(k, "max_steps")) c->max_steps = (uint32_t)atoi(v);
        else if (!strcmp(k, "prof")) c->prof = (uint32_t)atoi(v);
        else if (!strcmp(k, "prompt_tail")) c->prompt_tail = (uint32_t)atoi(v);
        else if (!strcmp(k, "lr")) c->lr = (float)atof(v);
        else if (!strcmp(k, "clip")) c->clip = (float)atof(v);
        else if (!strcmp(k, "init_std")) c->init_std = (float)atof(v);
        else if (!strcmp(k, "temp")) c->temp = (float)atof(v);
        else if (!strcmp(k, "markov")) c->markov = (uint32_t)atoi(v);
        else if (!strcmp(k, "lr_markov")) c->lr_markov = (float)atof(v);
        else if (!strcmp(k, "loss")) { if (!strcmp(v, "kl")) c->loss_kl = 1; else if (!strcmp(v, "tv")) c->loss_kl = 0; else { fprintf(stderr, "ds4: --draft-train loss 只认 tv/kl\n"); free(b); return false; } }
        else { fprintf(stderr, "ds4: --draft-train 不认的配置项 %s\n", k); free(b); return false; }
    }
    free(b);
    if (!c->texts[0] || !c->out[0]) { fprintf(stderr, "ds4: --draft-train 配置要有 texts= 与 out=\n"); return false; }
    if (!c->cache[0]) snprintf(c->cache, sizeof c->cache, "%s/cache", c->out);
    if (c->loss_kl && c->topk > 128u) { fprintf(stderr, "ds4: --draft-train loss=kl 的 topk %u 超 KL 核上限 128\n", c->topk); return false; }
    if (c->topk > 4096u) { fprintf(stderr, "ds4: --draft-train topk %u 超总变差核上限 4096\n", c->topk); return false; }
    if (!c->rank && !c->rank_exit && !c->markov) { fprintf(stderr, "ds4: --draft-train rank / rank_exit / markov 不能全是 0\n"); return false; }
    if (!(c->lr_markov > 0.f)) c->lr_markov = c->lr;
    return true;
}

/* 留出评估: 所有留出文本的全部块, 五位各自的陪审团三件 + KL。jury[B][3] 按位累加, jrows[B] 行数; 逐文本另打一行首位三件
 * (与 d1 accjury 同一份 ids 上的数直接对: 09-29 CFO 温 1.0 Σmin 0.7437 / 大盘 0.7574) */
static bool dk_eval(ds4_engine *e, const dk_cfg *c, dk_run *r, const dk_data *d, double *kl, double *jury, uint32_t *jrows) {
    const uint32_t B = r->B;
    double ls = 0.0; uint32_t nt = 0;
    memset(jury, 0, sizeof(double) * 3u * B); memset(jrows, 0, sizeof(uint32_t) * B);
    for (uint32_t i = 0; i < d->nt; i++) {
        const dk_text *t = &d->t[i];
        if (!t->eval) continue;
        double tj[DS4_MTP_MAX_BLOCK * 3]; uint32_t tr[DS4_MTP_MAX_BLOCK]; double tl = 0.0; uint32_t tn = 0;
        memset(tj, 0, sizeof tj); memset(tr, 0, sizeof tr);
        for (uint32_t i0 = t->b_lo; i0 < t->b_hi; i0 += c->batch_blocks) {
            const uint32_t nb = t->b_hi - i0 < c->batch_blocks ? t->b_hi - i0 : c->batch_blocks;
            if (!dk_batch(e, c, r, t, i0, nb, 0, &tl, &tn, tj, tr)) return false;
        }
        const double n1 = tr[0] ? (double)tr[0] : 1.0;
        fprintf(stderr, "ds4: [dk]   留出 %s: 损失 %.4f, 首位 Σmin %.4f / p(argmax q) %.4f / 贪心一致 %.4f (n=%u)\n", t->path, tn ? tl / tn : 0.0,
                tj[0] / n1, tj[1] / n1, tj[2] / n1, tr[0]);
        for (uint32_t j = 0; j < B; j++) { for (int q = 0; q < 3; q++) jury[j * 3 + q] += tj[j * 3 + q]; jrows[j] += tr[j]; }
        ls += tl; nt += tn;
    }
    *kl = nt ? ls / nt : 0.0;
    return true;
}
static void dk_jury_print(FILE *lf, const char *tag, double kl, const double *jury, const uint32_t *jrows, uint32_t B, float T) {
    double chain = 1.0, expect = 0.0;
    fprintf(stderr, "ds4: [dk] ★%s 留出: 损失 %.4f; 陪审团(温 %.2f, 逐位 Σmin / p(argmax q) / 贪心一致, n):", tag, kl, (double)T);
    if (lf) fprintf(lf, "%s eval_kl %.5f", tag, kl);
    for (uint32_t j = 0; j < B; j++) {
        const double n = jrows[j] ? (double)jrows[j] : 1.0, a = jury[j * 3] / n, b = jury[j * 3 + 1] / n, g = jury[j * 3 + 2] / n;
        chain *= a; expect += chain;
        fprintf(stderr, " 位%u %.4f/%.4f/%.4f(%u)", j + 1, a, b, g, jrows[j]);
        if (lf) fprintf(lf, " p%u %.5f %.5f %.5f", j + 1, a, b, g);
    }
    fprintf(stderr, "; 链乘积期望接受 %.3f/%u★\n", expect, B);
    if (lf) { fprintf(lf, " chain %.5f\n", expect); fflush(lf); }
}

/* 有限差分梯度检查(与 core_ptrain.c pt_gradcheck 同一方法): 第一个训练批, 每件的 A 沿 g/|g| 走 ±ε, (L+ − L−)/2ε 对 |g| */
static bool dk_gradcheck(ds4_engine *e, const dk_cfg *c, dk_run *r, const dk_data *d) {
    const dk_text *t = NULL;
    for (uint32_t i = 0; i < d->nt && !t; i++) if (!d->t[i].eval) t = &d->t[i];
    if (!t) { fprintf(stderr, "ds4: [dk 梯度检查] 没有训练文本\n"); return false; }
    const uint32_t nb = t->b_hi - t->b_lo < c->batch_blocks ? t->b_hi - t->b_lo : c->batch_blocks, E = DS4_N_EMBD;
    const uint32_t nit = r->NT + 1u;
    bool ok = true;
    for (uint32_t it = 0; ok && it < nit; it++) {
        ds4_gpu_tensor *A = it < r->NT ? r->st.ampA[it] : r->xA, *g = it < r->NT ? r->gA[it] : r->gxA;
        const uint32_t K = it < r->NT ? r->K : r->Kx;
        if (!A || !g || !K) continue;
        const uint64_t nel = (uint64_t)K * E;
        double l0 = 0; uint32_t n0 = 0;
        for (uint32_t T = 0; T < r->NT; T++) if (r->gA[T]) { ds4_gpu_tensor_fill_f32(r->gA[T], 0.f, (uint64_t)r->K * E); ds4_gpu_tensor_fill_f32(r->gB[T], 0.f, (uint64_t)r->K * E); }
        if (r->gxA) { ds4_gpu_tensor_fill_f32(r->gxA, 0.f, (uint64_t)r->Kx * E); ds4_gpu_tensor_fill_f32(r->gxB, 0.f, (uint64_t)r->Kx * E); }
        ok = dk_batch(e, c, r, t, t->b_lo, nb, 1, &l0, &n0, NULL, NULL);
        double ss = 0;
        if (ok) ok = ds4_gpu_bwd_sumsq_tensor(g, nel, &ss) != 0;
        const double gn = sqrt(ss);
        if (!ok || gn <= 0) { fprintf(stderr, "ds4: [dk 梯度检查] 件 %u |g| = %g, 查不了\n", it, gn); break; }
        ds4_gpu_tensor *D = v41_alloc(nel * 4, &ok);
        ok = ok && ds4_gpu_tensor_fill_f32(D, 0.f, nel) && ds4_gpu_bwd_axpy_tensor(D, g, (float)(1.0 / gn), nel);
        /* ε 从 0.1 扫到 10: 件的输出每塔出口都舍 bf16, 小 ε 的扰动大半掉进舍入死区(10-07 实撞: 塔件 ε=0.01/0.1/0.3/1 比值 0.01/0.09/0.26/0.71,
         * 逐档 ×3 ⇒ 不是反传错是扰动被吃掉), 只有扰动量级盖过 bf16 一格的档位有信号; ε 太大又进非线性区, 看中间几档比值是否 ≈ 1 */
        const float eps[5] = { 1e-1f, 3e-1f, 1.f, 3.f, 10.f };
        /* 扰动的两趟走冻结选择前向(各塔路由照上面基准那趟存的选): 反传把选择当常量, 量的也该是同一个函数 —— 不冻的话扰动塔 0 的件会翻
         * 塔 1/2 的并列路由边, 一次 0.003~0.01 的损失跳变盖过 ε·|g|(10-07 实撞: 塔件比值 −0.03~3.1 乱跳, 出口件 0.99) */
        r->save.replay = 1;
        for (int q = 0; ok && q < 5; q++) {
            double lp = 0, lm = 0; uint32_t n1 = 0, n2 = 0;
            ok = ds4_gpu_bwd_axpy_tensor(A, D, eps[q], nel) && dk_batch(e, c, r, t, t->b_lo, nb, 0, &lp, &n1, NULL, NULL) &&
                 ds4_gpu_bwd_axpy_tensor(A, D, -2.f * eps[q], nel) && dk_batch(e, c, r, t, t->b_lo, nb, 0, &lm, &n2, NULL, NULL) &&
                 ds4_gpu_bwd_axpy_tensor(A, D, eps[q], nel);
            const double norm = 1.0 / (double)(nb * r->B), fd = (lp - lm) * norm / (2.0 * eps[q]);
            fprintf(stderr, "ds4: [dk 梯度检查] %s%u ε=%.0e: 有限差分 %.6g / 反传 %.6g = %.4f (L0 %.5f L+ %.5f L− %.5f)\n",
                    it < r->NT ? "塔件 T" : "出口件 x", it < r->NT ? it : 0u, (double)eps[q], fd, gn, fd / gn, l0 * norm, lp * norm, lm * norm);
        }
        r->save.replay = 0;
        if (D) ds4_gpu_tensor_free(D);
    }
    for (uint32_t T = 0; T < r->NT; T++) if (r->gA[T]) { ds4_gpu_tensor_fill_f32(r->gA[T], 0.f, (uint64_t)r->K * E); ds4_gpu_tensor_fill_f32(r->gB[T], 0.f, (uint64_t)r->K * E); }
    if (r->gxA) { ds4_gpu_tensor_fill_f32(r->gxA, 0.f, (uint64_t)r->Kx * E); ds4_gpu_tensor_fill_f32(r->gxB, 0.f, (uint64_t)r->Kx * E); }
    return ok;
}

typedef struct { uint32_t text, i0, nb; } dk_batch_ref;

int ds4_engine_draft_train(ds4_engine *e, const char *spec) {
    if (!e || !spec || !ds4_engine_is_v41(e) || !e->metal_ready) { fprintf(stderr, "ds4: --draft-train 要 V4.1 模型 + GPU 后端\n"); return 1; }
    dk_cfg c;
    if (!dk_cfg_load(&c, spec)) return 1;
    mkdir(c.out, 0755);
    dk_data d;
    if (!dk_data_load(e, &c, &d)) { dk_data_free(&d); return 1; }
    const uint64_t rel = ds4_gpu_v41_scratch_release();
    fprintf(stderr, "ds4: [dk] 取料后放掉暂存 %.0f MB\n", (double)rel / 1048576.0);
    dk_run r;
    if (!dk_run_alloc(e, &c, &r)) { dk_data_free(&d); return 1; }
    const uint32_t B = r.B, E = DS4_N_EMBD;
    double jury[DS4_MTP_MAX_BLOCK * 3]; uint32_t jrows[DS4_MTP_MAX_BLOCK];
    double kl0 = 0, kl = 0;
    char lp[1100]; snprintf(lp, sizeof lp, "%s/train.log", c.out);
    FILE *lf = fopen(lp, "a");
    int rc = 1;
    /* 训练批清单: 每份训练文本按 batch_blocks 切连续块, 每轮打乱批序(种子固定) */
    uint32_t nbat = 0, cap = 64; dk_batch_ref *bats = xmalloc((size_t)cap * sizeof *bats);
    for (uint32_t i = 0; i < d.nt; i++) {
        const dk_text *t = &d.t[i];
        if (t->eval) continue;
        for (uint32_t i0 = t->b_lo; i0 < t->b_hi; i0 += c.batch_blocks) {
            if (nbat == cap) { cap *= 2; bats = realloc(bats, (size_t)cap * sizeof *bats); }
            bats[nbat++] = (dk_batch_ref){ i, i0, t->b_hi - i0 < c.batch_blocks ? t->b_hi - i0 : c.batch_blocks };
        }
    }
    fprintf(stderr, "ds4: [dk] 训练 %u 批(每批 ≤ %u 块 × %u 位), 塔件 K=%u 出口件 K=%u%s, lr %.2g, %u 轮, 目标 %s(教师 top-%u, 温 %.2f)\n", nbat, c.batch_blocks, B, r.K, r.Kx,
            r.mkE ? " + markov 偏置表" : "", (double)c.lr, c.epochs, c.loss_kl ? "前向 KL" : "总变差", c.topk, (double)c.temp);
    if (c.gradcheck) {
        if (!dk_gradcheck(e, &c, &r, &d)) goto out;
        if (c.gradcheck >= 2) { rc = 0; goto out; }
    }
    if (!c.max_steps) {
        if (!dk_eval(e, &c, &r, &d, &kl0, jury, jrows)) goto out;
        dk_jury_print(lf, c.init[0] ? "第 0 步(起点 = init 的件)" : "第 0 步(部署态草稿器)", kl0, jury, jrows, B, c.temp);
    }
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ c.seed;
    uint32_t step = 0;
    const double t0 = now_sec();
    for (uint32_t ep = 1; ep <= c.epochs; ep++) {
        for (uint32_t i = nbat; i > 1; i--) {   /* Fisher–Yates */
            rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
            const uint32_t j = (uint32_t)((rs * 2685821657736338717ull) % i);
            const dk_batch_ref t = bats[i - 1]; bats[i - 1] = bats[j]; bats[j] = t;
        }
        double esum = 0; uint32_t erows = 0;
        for (uint32_t bi = 0; bi < nbat; bi++) {
            const dk_batch_ref *bb = &bats[bi];
            double ls = 0; uint32_t nt = 0;
            if (!dk_batch(e, &c, &r, &d.t[bb->text], bb->i0, bb->nb, 1, &ls, &nt, NULL, NULL)) goto out;
            double ss = 0, a = 0;
            for (uint32_t T = 0; T < r.NT; T++) if (r.gA[T]) {
                if (!ds4_gpu_bwd_sumsq_tensor(r.gA[T], (uint64_t)r.K * E, &a)) goto out; ss += a;
                if (!ds4_gpu_bwd_sumsq_tensor(r.gB[T], (uint64_t)r.K * E, &a)) goto out; ss += a;
            }
            if (r.gxA) { if (!ds4_gpu_bwd_sumsq_tensor(r.gxA, (uint64_t)r.Kx * E, &a)) goto out; ss += a; if (!ds4_gpu_bwd_sumsq_tensor(r.gxB, (uint64_t)r.Kx * E, &a)) goto out; ss += a; }
            if (r.mkE) { if (!ds4_gpu_bwd_sumsq_tensor(r.gmkE, (uint64_t)r.Vm * r.Rk, &a)) goto out; ss += a; if (!ds4_gpu_bwd_sumsq_tensor(r.gmkH, (uint64_t)DS4_N_VOCAB * r.Rk, &a)) goto out; ss += a; }
            const double gn = sqrt(ss);
            const float gs = (c.clip > 0.f && gn > c.clip) ? (float)(c.clip / gn) : 1.0f;
            step++;
            for (uint32_t T = 0; T < r.NT; T++) if (r.gA[T] &&
                (!ds4_gpu_bwd_adam_tensor(r.st.ampA[T], r.gA[T], r.mA[T], r.vA[T], (uint64_t)r.K * E, c.lr, 0.9f, 0.999f, 1e-8f, gs, step) ||
                 !ds4_gpu_bwd_adam_tensor(r.st.ampB[T], r.gB[T], r.mB[T], r.vB[T], (uint64_t)r.K * E, c.lr, 0.9f, 0.999f, 1e-8f, gs, step))) goto out;
            if (r.gxA && (!ds4_gpu_bwd_adam_tensor(r.xA, r.gxA, r.mxA, r.vxA, (uint64_t)r.Kx * E, c.lr, 0.9f, 0.999f, 1e-8f, gs, step) ||
                          !ds4_gpu_bwd_adam_tensor(r.xB, r.gxB, r.mxB, r.vxB, (uint64_t)r.Kx * E, c.lr, 0.9f, 0.999f, 1e-8f, gs, step))) goto out;
            if (r.mkE && (!ds4_gpu_bwd_adam_tensor(r.mkE, r.gmkE, r.mE, r.vE, (uint64_t)r.Vm * r.Rk, c.lr_markov, 0.9f, 0.999f, 1e-8f, gs, step) ||
                          !ds4_gpu_bwd_adam_tensor(r.mkH, r.gmkH, r.mH, r.vH, (uint64_t)DS4_N_VOCAB * r.Rk, c.lr_markov, 0.9f, 0.999f, 1e-8f, gs, step))) goto out;
            esum += ls; erows += nt;
            if (lf) { fprintf(lf, "step %u ep %u loss %.5f gnorm %.4g t %.0f\n", step, ep, nt ? ls / nt : 0.0, gn, now_sec() - t0); fflush(lf); }
            if (step % 10 == 0 || bi + 1 == nbat)
                fprintf(stderr, "ds4: [dk] 轮 %u 步 %u/%u: 批损失 %.4f, 梯度范数 %.3g%s, %.0f s\n", ep, step, nbat * (ep - 1) + bi + 1, nt ? ls / nt : 0.0, gn, gs < 1.f ? "(裁剪)" : "", now_sec() - t0);
            if (c.max_steps && step >= c.max_steps) {
                if (c.prof && r.tm_n) fprintf(stderr, "ds4: [dk prof] %u 批(秒/批): 窗口 %.3f | 塔前向 %.3f | 出口 %.3f | 出口反传 %.3f | 塔反传 %.3f | 整批 %.3f\n", r.tm_n,
                                              r.tm[DK_TM_WIN] / r.tm_n, r.tm[DK_TM_TOWERS] / r.tm_n, r.tm[DK_TM_EXIT] / r.tm_n, r.tm[DK_TM_EXITB] / r.tm_n, r.tm[DK_TM_TOWERB] / r.tm_n, r.tm[DK_TM_BATCH] / r.tm_n);
                fprintf(stderr, "ds4: [dk] max_steps=%u 到了: 计时模式退出(不评估不存盘)\n", c.max_steps);
                rc = 0; goto out;
            }
        }
        if (!dk_eval(e, &c, &r, &d, &kl, jury, jrows)) goto out;
        char tag[48]; snprintf(tag, sizeof tag, "轮 %u 末(训练损失 %.4f)", ep, erows ? esum / erows : 0.0);
        dk_jury_print(lf, tag, kl, jury, jrows, B, c.temp);
        char ck[1100]; snprintf(ck, sizeof ck, "%s/ckpt_e%02u", c.out, ep);
        if (!dk_save(&c, &r, ck)) goto out;
    }
    if (c.epochs && !dk_save(&c, &r, c.out)) goto out;
    if (c.epochs) fprintf(stderr, "ds4: [dk] 件已落 %s(tower_Tn.bin + exit.dspa + base.fnv); 挂法: --draft-amp %s\n", c.out, c.out);
    rc = 0;
out:
    if (lf) fclose(lf);
    free(bats);
    dk_run_free(&r);
    dk_data_free(&d);
    return rc;
}
#else
int ds4_engine_draft_train(ds4_engine *e, const char *spec) { (void)e; (void)spec; return 1; }
#endif /* !DS4_NO_GPU */
