/* core_score_aux.c — 逐位 NLL / 逐位 top-K 出口。契约与文件格式见 core_score_aux.h。
 * 数值只有一处: log_softmax 走 max 平移(logits 绝对值能到几十, 直接 expf 会溢出),
 * NLL 与 top-K 的概率用同一个 (mx, se), 免得两个出口的口径各漂各的。 */
#include "core_internal.h"
#include "core_score_aux.h"

struct ds4_score_aux {
    FILE     *nf, *tf, *rf;
    int       topk;
    uint32_t  n, vocab;
    const char *tag;
    float    *tk_p;
    int      *tk_id;
    double    nll_sum, tk_mass_sum, rms_sum;
    uint32_t  nll_cnt, tk_rows, rms_rows;
};

static FILE *aux_open_or_die(const char *path, const char *tag, const char *what) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ds4: [%s] %s 打不开 %s -- aborting\n", tag, what, path); exit(1); }
    return f;
}

ds4_score_aux *ds4_score_aux_open(const char *nll_path, const char *topk_path, int topk,
                                  const char *rms_path, uint32_t n, uint32_t vocab, const char *tag) {
    const int want_nll = nll_path && nll_path[0];
    const int want_tk = topk > 0 && topk_path && topk_path[0];
    const int want_rms = rms_path && rms_path[0];
    if (!want_nll && !want_tk && !want_rms) return NULL;
    ds4_score_aux *a = xmalloc(sizeof *a);
    memset(a, 0, sizeof *a);
    a->topk = want_tk ? topk : 0;
    a->n = n; a->vocab = vocab; a->tag = tag ? tag : "score";
    if (want_nll) a->nf = aux_open_or_die(nll_path, a->tag, "nll");
    if (want_rms) a->rf = aux_open_or_die(rms_path, a->tag, "rms");
    if (want_tk) {
        a->tf = aux_open_or_die(topk_path, a->tag, "topk");
        /* 头带 K/S/VOCAB: 解算侧靠它自证读的是同一趟的产物, 不用另传参数 */
        const uint32_t hd[4] = { 0x44475445u /* "ETGD" */, (uint32_t)topk, n, vocab };
        if (fwrite(hd, sizeof(uint32_t), 4, a->tf) != 4) {
            fprintf(stderr, "ds4: [%s] topk 头短写 -- aborting\n", a->tag); exit(1);
        }
        a->tk_p = xmalloc((size_t)topk * sizeof(float));
        a->tk_id = xmalloc((size_t)topk * sizeof(int));
    }
    return a;
}

void ds4_score_aux_row(ds4_score_aux *a, uint32_t i, const float *lg, int tgt) {
    if (!a || !lg) return;
    const uint32_t vocab = a->vocab;
    float mx = lg[0];
    for (uint32_t k = 1; k < vocab; k++) if (lg[k] > mx) mx = lg[k];
    double se = 0.0;
    for (uint32_t k = 0; k < vocab; k++) se += exp((double)(lg[k] - mx));
    if (a->nf) {
        /* 位置 i 的 logits 预测 ids[i+1] —— 这个"错一格"是 teacher-forced 打分的全部要害,
         * 下游判决行也按它左移一格。NLL = logsumexp − lg[tgt], 与 anchor_metrics 同式。 */
        float v = NAN;
        if (tgt >= 0) {
            v = (float)(log(se) - (double)(lg[tgt] - mx));
            a->nll_sum += v; a->nll_cnt++;
        }
        if (fwrite(&v, sizeof(float), 1, a->nf) != 1) {
            fprintf(stderr, "ds4: [%s] nll 短写 @pos %u -- aborting\n", a->tag, i); exit(1);
        }
    }
    if (a->tf) {
        /* top-K 选择: K 槽最小值替换, 但★先用门槛挡一道★。朴素写法对每个词表项都扫一遍 K 槽找最小,
         * 是 O(V·K) = 12.9万×64 ≈ 830 万次比较/行, 5000 行的一趟光这一步就要一分多钟(实测打分时间
         * 里它占了大头)。维护"当前第 K 大的值"当门槛后, 绝大多数项一次比较就被挡掉, 只有真正能进
         * 榜的(期望 K·ln(V/K) ≈ 490 个)才去扫槽 —— 结果逐位相同, 只是不做无用功。 */
        const int K = a->topk;
        int nk = 0, mi = 0;
        float thr = -INFINITY;
        for (uint32_t k = 0; k < vocab; k++) {
            if (nk < K) {
                a->tk_id[nk] = (int)k; a->tk_p[nk] = lg[k]; nk++;
                if (nk == K) { mi = 0; for (int q = 1; q < K; q++) if (a->tk_p[q] < a->tk_p[mi]) mi = q; thr = a->tk_p[mi]; }
                continue;
            }
            if (lg[k] <= thr) continue;
            a->tk_p[mi] = lg[k]; a->tk_id[mi] = (int)k;
            mi = 0;
            for (int q = 1; q < K; q++) if (a->tk_p[q] < a->tk_p[mi]) mi = q;
            thr = a->tk_p[mi];
        }
        double mass = 0.0;
        for (int q = 0; q < nk; q++) {
            a->tk_p[q] = (float)(exp((double)(a->tk_p[q] - mx)) / se);
            mass += a->tk_p[q];
        }
        const float tgt_p = tgt >= 0 ? (float)(exp((double)(lg[tgt] - mx)) / se) : 0.0f;
        const float massf = (float)mass;
        a->tk_mass_sum += mass; a->tk_rows++;
        if (fwrite(&i, sizeof(uint32_t), 1, a->tf) != 1 ||
            fwrite(&tgt, sizeof(int), 1, a->tf) != 1 ||
            fwrite(&tgt_p, sizeof(float), 1, a->tf) != 1 ||
            fwrite(&massf, sizeof(float), 1, a->tf) != 1 ||
            fwrite(a->tk_id, sizeof(int), (size_t)nk, a->tf) != (size_t)nk ||
            fwrite(a->tk_p, sizeof(float), (size_t)nk, a->tf) != (size_t)nk) {
            fprintf(stderr, "ds4: [%s] topk 短写 @pos %u -- aborting\n", a->tag, i); exit(1);
        }
    }
}

void ds4_score_aux_rms_rows(ds4_score_aux *a, uint32_t i0, const float *x,
                            uint32_t nrow, uint32_t dim, float eps) {
    if (!a || !a->rf || !x) return;
    for (uint32_t r = 0; r < nrow; r++) {
        const float *xr = x + (size_t)r * dim;
        /* f32 累加(与 kernel 逐位同): 换成 double 会和落地那条路差个几 ulp, 而这个数是要拿去
         * 预测 logit 差的, 两边口径必须是同一个。 */
        float s = 0.f;
        for (uint32_t i = 0; i < dim; i++) s += xr[i] * xr[i];
        const float inv = 1.0f / sqrtf(s / (float)dim + eps);
        a->rms_sum += inv; a->rms_rows++;
        if (fwrite(&inv, sizeof(float), 1, a->rf) != 1) {
            fprintf(stderr, "ds4: [%s] rms 短写 @pos %u -- aborting\n", a->tag, i0 + r); exit(1);
        }
    }
}

/* 占位行(2026-09-23, 部署同路打分): CED 块只跑编码器段, 没有 logits/出口隐态, 但三份表都是按行顺序写、下游按行号读 ——
 * 跳过不写 = 后面整体错位(读侧会拦: "第 0 条记的行号是 9119")。占位行: NLL = NaN, top-K 行号照写、tgt = −1、概率全 0、
 * id 全 −1(读侧 p>0 才认), rms = 0。不计入平均值, 冒烟读数只反映真算过的行。 */
void ds4_score_aux_skip_rows(ds4_score_aux *a, uint32_t i0, uint32_t nrow) {
    if (!a) return;
    const float nanv = NAN, zero = 0.0f;
    const int neg = -1;
    for (uint32_t r = 0; r < nrow; r++) {
        const uint32_t i = i0 + r;
        if (a->nf && fwrite(&nanv, sizeof(float), 1, a->nf) != 1) { fprintf(stderr, "ds4: [%s] nll 短写 @pos %u -- aborting\n", a->tag, i); exit(1); }
        if (a->rf && fwrite(&zero, sizeof(float), 1, a->rf) != 1) { fprintf(stderr, "ds4: [%s] rms 短写 @pos %u -- aborting\n", a->tag, i); exit(1); }
        if (a->tf) {
            bool ok = fwrite(&i, sizeof(uint32_t), 1, a->tf) == 1 && fwrite(&neg, sizeof(int), 1, a->tf) == 1 &&
                      fwrite(&zero, sizeof(float), 1, a->tf) == 1 && fwrite(&zero, sizeof(float), 1, a->tf) == 1;
            for (int q = 0; ok && q < a->topk; q++) ok = fwrite(&neg, sizeof(int), 1, a->tf) == 1;
            for (int q = 0; ok && q < a->topk; q++) ok = fwrite(&zero, sizeof(float), 1, a->tf) == 1;
            if (!ok) { fprintf(stderr, "ds4: [%s] topk 短写 @pos %u -- aborting\n", a->tag, i); exit(1); }
        }
    }
}

void ds4_score_aux_close(ds4_score_aux *a) {
    if (!a) return;
    if (a->nf) {
        if (fclose(a->nf) != 0) { fprintf(stderr, "ds4: [%s] 关闭 nll 文件失败 -- aborting\n", a->tag); exit(1); }
        /* 全序列平均只是冒烟读数(判决要按 --rows 取报告段/分歧段)。打出来是为了跑完立刻
         * 能看出"这趟是不是废的" —— 数量级不对就不用往下走了。 */
        fprintf(stderr, "ds4: [%s] NLL 已写: n=%u 全序列平均 %.4f (PPL %.4f)\n", a->tag,
                a->nll_cnt, a->nll_cnt ? a->nll_sum / a->nll_cnt : 0.0,
                a->nll_cnt ? exp(a->nll_sum / a->nll_cnt) : 0.0);
    }
    if (a->tf) {
        if (fclose(a->tf) != 0) { fprintf(stderr, "ds4: [%s] 关闭 topk 文件失败 -- aborting\n", a->tag); exit(1); }
        /* ★平均覆盖质量是这条近似路的自证★: 它低了就说明 top-K 截得太狠, 靶里丢掉的那部分
         * 梯度不是小量, 该把 K 调大而不是硬解。 */
        fprintf(stderr, "ds4: [%s] topK=%d 已写: %u 行, 平均覆盖概率质量 %.4f\n", a->tag,
                a->topk, a->tk_rows, a->tk_rows ? a->tk_mass_sum / a->tk_rows : 0.0);
    }
    if (a->rf) {
        if (fclose(a->rf) != 0) { fprintf(stderr, "ds4: [%s] 关闭 rms 文件失败 -- aborting\n", a->tag); exit(1); }
        /* 行数必须等于 S: 少了就是某块没写(钩子提前结束那种), 下游按行号取值会整体错位 */
        fprintf(stderr, "ds4: [%s] rms(inv) 已写: %u/%u 行, 平均 inv %.6f\n", a->tag,
                a->rms_rows, a->n, a->rms_rows ? a->rms_sum / a->rms_rows : 0.0);
    }
    free(a->tk_p); free(a->tk_id); free(a);
}
