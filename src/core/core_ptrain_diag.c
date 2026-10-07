/* core_ptrain_diag.c — 后训练 ③ 第八版的逐位诊断: 配置 diag=1(2026-10-02)。总述见 core_ptrain.h。
 *
 * 为什么要它: 选轮用的留出 KL 是整段答案的平均, 量不到"答二还是答三"那一位 —— 10-02 1+1=3 那趟留出 KL 降了 80%,
 * 同样的问法贪心照样答二。这里对清单里每道题的答案逐位并排打三份分布: 教师(读块) / 学生挂 ③(init= 那份) / 部署态(③ 置零),
 * 各取前 PT_DIAG_SHOW 名, 外加参考 token 在三份里的概率。答案可以填学生自己的贪心输出, 打到的就是它自己走的那条路上的决策位。
 * 只读不训: 不预热、不训练、不落 ③; 结果写 <out>/diag.txt, stderr 每题只打"教师与部署态榜首第一次分叉"那一位。
 * 出错会怎样: 不给 init= 时"挂 ③"那一列就是 A=0 的部署态, 两列逐位相同 —— 不是 bug, 是没挂东西。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define PT_DIAG_SHOW 5u   /* 每份分布打前几名; top-K 核是逐轮取最大, 出来就是降序 */

typedef struct { int32_t *id; float *p, *rest; } pt_dtab;   /* 一道题的学生分布 [m][K] / [m][K] / [m] */

static void pt_dtab_free(pt_dtab *t) { free(t->id); free(t->p); free(t->rest); memset(t, 0, sizeof *t); }

/* 学生前向(状态里挂着什么 ③ 就是什么) → 答案位 top-K, 与教师表同一个核、同一个 K */
static bool pt_diag_student(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s, pt_dtab *o) {
    ds4_v41_state *st = &r->st;
    const uint32_t TK = c->topk, m = s->m;
    pt_state_reset(st);
    st->ced_skip = 0; st->head_last_only = 0;
    if (!v41_forward(e, st, s->sids, s->sn)) return false;
    if (!ds4_gpu_bwd_topk_tensor(r->tid, r->tp, r->trest, st->logits, s->sa0 - 1u, m, DS4_N_VOCAB, TK)) return false;
    const size_t mk = (size_t)m * TK;
    o->id = xmalloc(mk * 4); o->p = xmalloc(mk * 4); o->rest = xmalloc((size_t)m * 4);
    return ds4_gpu_tensor_read(r->tid, 0, o->id, mk * 4) && ds4_gpu_tensor_read(r->tp, 0, o->p, mk * 4) &&
           ds4_gpu_tensor_read(r->trest, 0, o->rest, (uint64_t)m * 4);
}

static void pt_diag_tok(FILE *f, ds4_engine *e, int32_t id) {
    if (id == ds4_token_eos(e)) { fputs("<EOS>", f); return; }
    size_t len = 0;
    char *t = ds4_token_text(e, id, &len);
    fputs("「", f);
    for (size_t i = 0; t && i < len; i++) { if (t[i] == '\n') fputs("\\n", f); else fputc(t[i], f); }
    fputs("」", f);
    free(t);
}

static int pt_diag_isnum(ds4_engine *e, int32_t id) {   /* token 文本含阿拉伯数字: 复盘里价格/日期/家数这些"要记住的事实"位 */
    size_t len = 0;
    char *t = ds4_token_text(e, id, &len);
    int has = 0;
    for (size_t i = 0; t && i < len && !has; i++) has = t[i] >= '0' && t[i] <= '9';
    free(t);
    return has;
}

static float pt_diag_pof(const int32_t *id, const float *p, uint32_t K, int32_t want) {   /* 榜外按 0 计(真值 < 余量, 汇总里是下界) */
    for (uint32_t k = 0; k < K; k++) if (id[k] == want) return p[k];
    return 0.f;
}

/* 汇总: 全部位 + 分叉位(教师榜首 ≠ 部署态榜首 = 材料改了模型的选择)。conf = 只算教师榜首 p ≥ 0.5 的分叉位(风格类小分歧剔掉) */
typedef struct { uint64_t n, t_ref, s_t; double pt, ps, pd; } pt_dsum;
static void pt_dsum_add(pt_dsum *a, int t_is_ref, int s_is_t, float pt, float ps, float pd) {
    a->n++; a->t_ref += (uint64_t)t_is_ref; a->s_t += (uint64_t)s_is_t; a->pt += pt; a->ps += ps; a->pd += pd;
}
static void pt_dsum_print(FILE *f, const char *name, const pt_dsum *a) {
    if (!a->n) { fprintf(f, "%s: 0 位\n", name); return; }
    const double n = (double)a->n;
    fprintf(f, "%s: %llu 位 | 教师榜首 = 参考答案 %.1f%% | 挂③榜首 = 教师榜首 %.1f%% | 教师榜首那个 token 的平均概率: 教师 %.3f / 挂③ %.3f / 部署 %.3f\n",
            name, (unsigned long long)a->n, 100.0 * (double)a->t_ref / n, 100.0 * (double)a->s_t / n, a->pt / n, a->ps / n, a->pd / n);
}

/* 一列: 参考 token 的概率(榜外 = 不在前 K, 只知道 < 余量) + 前 PT_DIAG_SHOW 名 */
static void pt_diag_col(FILE *f, ds4_engine *e, const char *name, const int32_t *id, const float *p, float rest, uint32_t K, int32_t ref) {
    int hit = -1;
    for (uint32_t k = 0; k < K && hit < 0; k++) if (id[k] == ref) hit = (int)k;
    if (hit >= 0) fprintf(f, " | %s 参考 %.3f:", name, p[hit]);
    else fprintf(f, " | %s 参考 榜外(<%.3f):", name, rest);
    for (uint32_t k = 0; k < PT_DIAG_SHOW && k < K; k++) { fputc(' ', f); pt_diag_tok(f, e, id[k]); fprintf(f, "%.3f", p[k]); }
}

bool pt_diag(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    const uint32_t TK = c->topk;
    char path[1200]; snprintf(path, sizeof path, "%s/diag.txt", c->out);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ds4: [ptrain diag] 写不了 %s\n", path); return false; }
    pt_dtab *with = xmalloc_zeroed(d->ns, sizeof *with), *bare = xmalloc_zeroed(d->ns, sizeof *bare);
    bool ok = true;
    /* 先挂 ③ 跑一遍, 再把 A 置零(= 部署态, 放大器输出 xn·B·A 恰为 0)跑第二遍 —— 置零不可逆, 顺序不能反 */
    for (uint32_t i = 0; ok && i < d->ns; i++)
        if (d->s[i].top_id && d->s[i].sn <= r->st.cap_tok) ok = pt_diag_student(e, c, r, &d->s[i], &with[i]);
    for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++) ok = ds4_gpu_tensor_fill_f32(r->st.ampA[il], 0.f, (uint64_t)r->K * DS4_N_EMBD);
    for (uint32_t i = 0; ok && i < d->ns; i++) if (with[i].id) ok = pt_diag_student(e, c, r, &d->s[i], &bare[i]);
    if (!ok) fprintf(stderr, "ds4: [ptrain diag] 学生前向失败\n");
    fprintf(f, "# 每行 = 答案的一位: 参考 token | 教师(读块)| 挂③(%s)| 部署态(③ 置零); 每列先给参考 token 的概率, 再给前 %u 名\n\n",
            c->init[0] ? c->init : "未给 init, 与部署态相同", PT_DIAG_SHOW);
    pt_dsum all = {0}, frk = {0}, frk_c = {0};
    uint64_t d_t = 0;                /* 全部位里部署态榜首 = 教师榜首的位数(对照: 挂 ③ 之前就一致的有多少) */
    double rt = 0, rs = 0, rd = 0;   /* 参考 token 的平均概率(榜外按 0) */
    uint64_t nn = 0, nn_t = 0, nn_s = 0, nn_d = 0, nnf = 0, nnf_s = 0;   /* 含数字的参考位: 位数 / 三方榜首 = 参考 / 其中分叉位与挂 ③ 跟上 */
    double nrt = 0, nrs = 0, nrd = 0;
    for (uint32_t i = 0; ok && i < d->ns; i++) {
        const pt_sample *s = &d->s[i];
        if (!with[i].id) continue;
        fprintf(f, "#%u 问: %s\n参考答案: %s\n", i, s->q, s->a);
        int fork = -1;   /* 教师与部署态榜首第一次不同的那一位 = 材料改变了模型选择的地方 */
        for (uint32_t j = 0; j < s->m; j++) {
            const int32_t ref = s->sids[s->sa0 + j];
            const size_t o = (size_t)j * TK;
            fprintf(f, "  [%u] 参考 ", j); pt_diag_tok(f, e, ref);
            pt_diag_col(f, e, "教师", s->top_id + o, s->top_p + o, s->top_rest[j], TK, ref);
            pt_diag_col(f, e, "挂③", with[i].id + o, with[i].p + o, with[i].rest[j], TK, ref);
            pt_diag_col(f, e, "部署", bare[i].id + o, bare[i].p + o, bare[i].rest[j], TK, ref);
            fputc('\n', f);
            if (fork < 0 && s->top_id[o] != bare[i].id[o]) fork = (int)j;
            const int32_t T = s->top_id[o];
            const float pt = s->top_p[o], ps = pt_diag_pof(with[i].id + o, with[i].p + o, TK, T), pd = pt_diag_pof(bare[i].id + o, bare[i].p + o, TK, T);
            const int t_is_ref = T == ref, s_is_t = with[i].id[o] == T;
            pt_dsum_add(&all, t_is_ref, s_is_t, pt, ps, pd);
            d_t += bare[i].id[o] == T;
            rt += pt_diag_pof(s->top_id + o, s->top_p + o, TK, ref);
            rs += pt_diag_pof(with[i].id + o, with[i].p + o, TK, ref);
            rd += pt_diag_pof(bare[i].id + o, bare[i].p + o, TK, ref);
            if (bare[i].id[o] != T) {
                pt_dsum_add(&frk, t_is_ref, s_is_t, pt, ps, pd);
                if (pt >= 0.5f) pt_dsum_add(&frk_c, t_is_ref, s_is_t, pt, ps, pd);
            }
            if (pt_diag_isnum(e, ref)) {
                nn++; nn_t += t_is_ref; nn_s += with[i].id[o] == ref; nn_d += bare[i].id[o] == ref;
                nrt += pt_diag_pof(s->top_id + o, s->top_p + o, TK, ref);
                nrs += pt_diag_pof(with[i].id + o, with[i].p + o, TK, ref);
                nrd += pt_diag_pof(bare[i].id + o, bare[i].p + o, TK, ref);
                if (bare[i].id[o] != T) { nnf++; nnf_s += s_is_t; }
            }
        }
        fputc('\n', f);
        if (fork >= 0) {
            const size_t o = (size_t)fork * TK;
            fprintf(stderr, "ds4: [ptrain diag] #%u %s → 第 %d 位分叉: 教师 ", i, s->q, fork); pt_diag_tok(stderr, e, s->top_id[o]);
            fprintf(stderr, "%.3f / 挂③ ", s->top_p[o]); pt_diag_tok(stderr, e, with[i].id[o]);
            fprintf(stderr, "%.3f / 部署 ", with[i].p[o]); pt_diag_tok(stderr, e, bare[i].id[o]);
            fprintf(stderr, "%.3f\n", bare[i].p[o]);
        } else fprintf(stderr, "ds4: [ptrain diag] #%u %s → 教师与部署态榜首逐位相同\n", i, s->q);
    }
    /* 汇总写进文件末尾并打到 stderr: 分叉位 = 材料把模型的选择改了的位置, 挂 ③ 跟上的比例 = 这份 ③ 把材料写进去了多少 */
    for (int w = 0; ok && w < 2; w++) {
        FILE *o = w ? stderr : f;
        const char *pre = w ? "ds4: [ptrain diag] " : "# ";
        fprintf(o, "%s汇总 —— 全部位 %llu, 其中部署态榜首 = 教师榜首 %.1f%%(挂 ③ 之前就一致)\n", pre, (unsigned long long)all.n, all.n ? 100.0 * (double)d_t / (double)all.n : 0.0);
        fprintf(o, "%s", pre); pt_dsum_print(o, "全部位", &all);
        fprintf(o, "%s", pre); pt_dsum_print(o, "分叉位(教师榜首≠部署榜首)", &frk);
        fprintf(o, "%s", pre); pt_dsum_print(o, "分叉位且教师榜首 p≥0.5", &frk_c);
        if (all.n) fprintf(o, "%s参考 token 平均概率(榜外按 0): 教师 %.3f / 挂③ %.3f / 部署 %.3f\n", pre, rt / (double)all.n, rs / (double)all.n, rd / (double)all.n);
        if (nn) fprintf(o, "%s含数字的参考位: %llu 位 | 榜首 = 参考: 教师 %.1f%% / 挂③ %.1f%% / 部署 %.1f%% | 参考平均概率: 教师 %.3f / 挂③ %.3f / 部署 %.3f | 其中分叉位 %llu, 挂③跟上教师 %.1f%%\n",
                        pre, (unsigned long long)nn, 100.0 * (double)nn_t / (double)nn, 100.0 * (double)nn_s / (double)nn, 100.0 * (double)nn_d / (double)nn,
                        nrt / (double)nn, nrs / (double)nn, nrd / (double)nn, (unsigned long long)nnf, nnf ? 100.0 * (double)nnf_s / (double)nnf : 0.0);
    }
    fclose(f);
    for (uint32_t i = 0; i < d->ns; i++) { pt_dtab_free(&with[i]); pt_dtab_free(&bare[i]); }
    free(with); free(bare);
    if (ok) fprintf(stderr, "ds4: [ptrain diag] 写好 %s\n", path);
    return ok;
}

/* ---- packcheck: 合批的门(=1)与 STE 尖峰普查(=2), 10-02 定版 ----
 * 为什么不拿"凑满一包 vs 单题逐题"对账: 行数一变 cuBLAS 换算法、末位舍入就变, 个别 token 的 MoE 路由在并列边上被翻, 前向轨迹分叉
 * (10-02 实测: 同一题, token 10 在 L20 入口 hc 模长单题 121.7 / 合批 255.6), 两路都是这个模型的合法算法, 逐题对不上不说明谁错。
 * 判对错只认有限差分(冻结路由): 反传对着本路自己的前向求导, 沿它自己的梯度方向走 ±ε, 比值 ≈ 1 才算对。 */
static void pt_grad_zero(const pt_cfg *c, pt_run *r) {
    for (uint32_t k = c->layer_lo; k <= c->layer_hi; k++) {
        (void)ds4_gpu_tensor_fill_f32(r->gA[k], 0.f, (uint64_t)r->K * DS4_N_EMBD); (void)ds4_gpu_tensor_fill_f32(r->gB[k], 0.f, (uint64_t)r->K * DS4_N_EMBD);
    }
}
/* 同一题跑一趟: pack = 0 单题路, 1 = 一包两份(同一题; 第二份从第 n 行起) */
static bool pt_fd_run(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s, int pack, int grad, double *loss) {
    double le[2] = { 0, 0 }; uint32_t nt[2] = { 0, 0 };
    const pt_sample *two[2] = { s, s };
    const bool ok = pack ? pt_pack_step(e, c, r, two, 2, grad, le, nt) : pt_step_sample(e, c, r, s, grad, &le[0], &nt[0]);
    *loss = le[0] + le[1];
    return ok;
}
/* 第 il 层放大器 A 沿本路梯度方向的有限差分比值(ε = 0.1 / 0.3 两档; 同 pt_gradcheck 的冻结选择前向) */
static bool pt_fd_ratio(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_sample *s, int pack, uint32_t il, ds4_gpu_tensor *D, double rt[2], double *gnorm) {
    const uint64_t nel = (uint64_t)r->K * DS4_N_EMBD;
    const double norm = 1.0 / ((double)s->m * (double)c->batch);
    const float eps[2] = { 1e-1f, 3e-1f };
    double l0 = 0, ss = 0;
    pt_grad_zero(c, r);
    bool ok = pt_fd_run(e, c, r, s, pack, 1, &l0) && ds4_gpu_bwd_sumsq_tensor(r->gA[il], nel, &ss);
    *gnorm = sqrt(ss);
    if (!ok || *gnorm <= 0) return false;
    ok = ds4_gpu_tensor_fill_f32(D, 0.f, nel) && ds4_gpu_bwd_axpy_tensor(D, r->gA[il], (float)(1.0 / *gnorm), nel);
    r->save.replay = 1;
    for (int q = 0; ok && q < 2; q++) {
        double lp = 0, lm = 0;
        ok = ds4_gpu_bwd_axpy_tensor(r->st.ampA[il], D, eps[q], nel) && pt_fd_run(e, c, r, s, pack, 0, &lp) &&
             ds4_gpu_bwd_axpy_tensor(r->st.ampA[il], D, -2.f * eps[q], nel) && pt_fd_run(e, c, r, s, pack, 0, &lm) &&
             ds4_gpu_bwd_axpy_tensor(r->st.ampA[il], D, eps[q], nel);
        rt[q] = (lp - lm) * norm / (2.0 * eps[q]) / *gnorm;
    }
    r->save.replay = 0;
    pt_grad_zero(c, r);
    return ok;
}

bool pt_pack_check(ds4_engine *e, const pt_cfg *c, pt_run *r, const pt_data *d) {
    const pt_sample *s = NULL;   /* 第一道非保持料训练题(同 pt_check_sample), 放得下两份 */
    for (uint32_t i = 0; i < d->ns && !s; i++)
        if (!d->s[i].eval && pt_runnable(r, &d->s[i]) && !d->ch[d->s[i].chunk].hold && 2u * d->s[i].sn <= r->st.cap_tok) s = &d->s[i];
    if (!s || !r->pk.ready) { fprintf(stderr, "ds4: [packcheck] 没有能放两份的非保持料训练题(cap %u)或合批没开\n", r->st.cap_tok); return false; }
    bool ok = true;
    ds4_gpu_tensor *D = v41_alloc((uint64_t)r->K * DS4_N_EMBD * 4, &ok);
    if (c->packcheck == 2u) {   /* 普查: 单题路在 L8 按题长扫有限差分 —— STE 尖峰(10-02: 52/64 行两道比值 0.06/0.63, 其余 0.87~1.14)多常见 */
        static const uint32_t want[] = { 40, 52, 64, 72, 96, 128, 160, 200, 300 };
        for (size_t w = 0; ok && w < sizeof want / sizeof want[0]; w++) {
            const pt_sample *t = NULL;
            for (uint32_t i = 0; i < d->ns; i++) {
                const pt_sample *u = &d->s[i];
                if (u->eval || !pt_runnable(r, u) || d->ch[u->chunk].hold) continue;
                if (!t || abs((int)u->sn - (int)want[w]) < abs((int)t->sn - (int)want[w])) t = u;
            }
            double rt[2], gn = 0;
            ok = t && pt_fd_ratio(e, c, r, t, 0, 8u, D, rt, &gn);
            if (ok) fprintf(stderr, "ds4: [packcheck 普查] 单题路 %u 行(答案 %u): L08 有限差分/反传 = %.4f(ε=0.1) %.4f(ε=0.3), |反传| %.4g\n", t->sn, t->m, rt[0], rt[1], gn);
        }
        if (D) ds4_gpu_tensor_free(D);
        return ok;
    }
    /* ① 结构门: 同一题单独成一包 = 行数与单题路相同(cuBLAS 选同一组算法) ⇒ 40 层入口 hc 存档逐位同、KL 逐位同、梯度范数差 ≤ 1%(原子加次序 ~0.6%) */
    const uint64_t hrow = (uint64_t)DS4_N_HC * DS4_N_EMBD, nel = (uint64_t)r->K * DS4_N_EMBD;
    uint16_t *keep = xmalloc((size_t)s->sn * DS4_N_LAYER * hrow * 2), *h = xmalloc((size_t)s->sn * hrow * 2);
    double l1 = 0, l2 = 0, g1[DS4_MAX_LAYER], g2[DS4_MAX_LAYER], g0[DS4_MAX_LAYER], gmax = 0, nmax = 0;
    uint32_t nt = 0, ndiff_layers = 0, imax = 0, inmax = 0;
    const pt_sample *one[1] = { s };
    {   /* 本底: 单题路自己先跑一遍(反传有原子加, 次序不定), 与下面第二遍单题路比 = 噪声地板 —— 门按"合批差不超过本底的 3 倍且 ≤ 1e-2" */
        double l0 = 0;
        pt_grad_zero(c, r);
        ok = pt_step_sample(e, c, r, s, 1, &l0, &nt);
        for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++) { double a = 0, b = 0; ok = ds4_gpu_bwd_sumsq_tensor(r->gA[il], nel, &a) && ds4_gpu_bwd_sumsq_tensor(r->gB[il], nel, &b); g0[il] = sqrt(a + b); }
    }
    pt_grad_zero(c, r);
    ok = ok && pt_step_sample(e, c, r, s, 1, &l1, &nt);
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++)
        ok = !r->save.hc_in[il] || ds4_gpu_tensor_read(r->save.hc_in[il], 0, keep + (size_t)il * s->sn * hrow, (uint64_t)s->sn * hrow * 2);
    for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++) { double a = 0, b = 0; ok = ds4_gpu_bwd_sumsq_tensor(r->gA[il], nel, &a) && ds4_gpu_bwd_sumsq_tensor(r->gB[il], nel, &b); g1[il] = sqrt(a + b); }
    pt_grad_zero(c, r);
    ok = ok && pt_pack_step(e, c, r, one, 1, 1, &l2, &nt);
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (!r->save.hc_in[il]) continue;
        ok = ds4_gpu_tensor_read(r->save.hc_in[il], 0, h, (uint64_t)s->sn * hrow * 2);
        if (ok && memcmp(h, keep + (size_t)il * s->sn * hrow, (size_t)s->sn * hrow * 2) != 0) ndiff_layers++;
    }
    for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++) {
        double a = 0, b = 0;
        ok = ds4_gpu_bwd_sumsq_tensor(r->gA[il], nel, &a) && ds4_gpu_bwd_sumsq_tensor(r->gB[il], nel, &b);
        g2[il] = sqrt(a + b);
        const double rd = fabs(g2[il] - g1[il]) / (g1[il] > 1e-30 ? g1[il] : 1e-30), rn = fabs(g0[il] - g1[il]) / (g1[il] > 1e-30 ? g1[il] : 1e-30);
        if (rd > gmax) { gmax = rd; imax = il; }
        if (rn > nmax) { nmax = rn; inmax = il; }
    }
    pt_grad_zero(c, r);
    free(keep); free(h);
    /* 判据按本底: 这道题若撞上 STE 尖峰(10-02: 低层梯度被一个 token 主导、放大几百倍), 原子加次序的噪声在低层能到 2%(单题自己两遍 2.07e-2 @L06) */
    const bool pass1 = ok && ndiff_layers == 0 && l1 == l2 && gmax <= fmax(1e-2, 3.0 * nmax);
    fprintf(stderr, "ds4: [packcheck ①结构] %u 行一题一包: 入口 hc 逐位不同的层 %u / 40, KL 和 %.6f vs %.6f, 梯度范数最大相对差 合批 %.2e(L%02u) / 单题自己两遍 %.2e(L%02u) ⇒ %s\n",
            s->sn, ndiff_layers, l1, l2, gmax, imax, nmax, inmax, pass1 ? "过" : "★没过★");
    /* ② 有限差分门: 同一题一包两份(第二份从第 n 行起、跨段的视图都用上), 末层 / 解码器段 / 编码器段 / 底层四层 */
    static const uint32_t ls[] = { 39, 20, 8, 2 };
    bool pass2 = ok;
    for (size_t q = 0; ok && q < sizeof ls / sizeof ls[0]; q++) {
        double rt[2], gn = 0;
        ok = pt_fd_ratio(e, c, r, s, 1, ls[q], D, rt, &gn);
        const double best = fabs(rt[0] - 1.0) < fabs(rt[1] - 1.0) ? rt[0] : rt[1];
        if (!(best >= 0.8 && best <= 1.25)) pass2 = false;
        if (ok) fprintf(stderr, "ds4: [packcheck ②有限差分] 一包两份 L%02u: 比值 %.4f(ε=0.1) %.4f(ε=0.3), |反传| %.4g\n", ls[q], rt[0], rt[1], gn);
    }
    if (D) ds4_gpu_tensor_free(D);
    fprintf(stderr, "ds4: [packcheck] %s(①结构 %s, ②有限差分 %s: 两档里较好的比值须在 [0.8, 1.25])\n", ok && pass1 && pass2 ? "过" : "★没过★",
            pass1 ? "过" : "没过", pass2 ? "过" : "没过");
    return ok && pass1 && pass2;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_diag_nonempty_tu;
