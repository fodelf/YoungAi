/* anchor_metrics.c — 锚(FP 参考)与学生 logits 的五指标判决器(C, 2026-08-23)。
 * 取代 scripts/anchor_metrics.py(铁律: 链上不留 Python)。指标口径逐式同源:
 *   PPL        exp(mean NLL), teacher-forced, 位置 i 预测 ids[i+1]
 *   Mean KLD   mean_i Σ_v p_ref·(ln p_ref − ln p_stu)
 *   RMS Δp     top-32 联合窗(ref∪stu 各 top32 的并集)内均方 → sqrt(mean)×100
 *              (2026-08-03 用户纠: 全词表均方被 12.9 万近零项冲稀无信息量)
 *   Same top   mean_i [argmax ref == argmax stu]
 *   Δp(top)    |p_ref(ref top) − p_stu(ref top)| 均值 + p95
 * 锚格式 DQA2: <8×u32 头> <8B idh> <跳 fin/ridx/rw/H> <f32 logits[S][VOCAB]>
 * 学生格式:   <i32 S><i32 V><f32 logits[S][V]>
 * 用法: anchor_metrics --ref anchor.bin --ids ids.txt [--student stu.bin]
 *       [--ref-raw raw.bin | --ref-eval eval.bin] [--fit N] [--tail N] [--rows a:b[,c:d,...]]
 * --ref-raw = --score-out(解码路, 带 <S,V> 头); --ref-eval = --eval-logits(prefill 批路, 无头带 BOS 行)
 * --ref-nll = --eval-nll 的 f32[S](引擎已算好逐位 NLL, 只出 PPL/NLL; 省掉 5 个数量级的盘)
 * ★--rows 收多段(2026-09-08)★: 后训练判决只算"本应写出的报告"那些 token, 每条样本的
 * 材料段不算数 ⇒ N 条样本 = N 段不连续行, 合成一个池子出一份平均 NLL(不是每段各出一份)。*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

typedef struct { int S, HCM, DIM, NL, VOCAB, NACT; } meta_t;

static float *read_anchor(const char *path, meta_t *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    uint32_t hd[8];
    if (fread(hd, 4, 8, f) != 8 || hd[0] != 0x32415144u) { fprintf(stderr, "%s: 不是 DQA2 锚\n", path); exit(2); }
    m->S = (int)hd[1]; m->HCM = (int)hd[2]; m->DIM = (int)hd[3];
    m->NL = (int)hd[4]; m->VOCAB = (int)hd[5]; m->NACT = (int)hd[6];
    fseek(f, 8, SEEK_CUR);   /* idh */
    long long skip = (long long)m->NL * m->S * m->DIM * 4
                   + (long long)m->NL * m->S * m->NACT * 4 * 2
                   + (long long)m->NL * m->S * m->HCM * m->DIM * 4;
    fseek(f, (long)skip, SEEK_CUR);
    size_t n = (size_t)m->S * m->VOCAB;
    float *lg = malloc(n * sizeof(float));
    if (!lg || fread(lg, 4, n, f) != n) { fprintf(stderr, "%s: logits 截断\n", path); exit(2); }
    fclose(f);
    return lg;
}

/* --eval-ids 批量路(prefill)的输出: 裸 f32 [n][V] 无头, 且流首插了 BOS ⇒ 比 ids 多一行。
 * V 由文件大小反推(rows = nids+1), 掐掉 BOS 行后取前 nids 行 —— 与 --stu-raw 同一口径。
 * 为什么要这条路: 逐 token 的 --score-ids 是解码路(~20 t/s), 一份 8000 字的报告就要 4 分钟;
 * 批路做同样的 teacher-forced 打分快一个量级, 长报告的判决只能走它。 */
static float *read_eval_raw(const char *path, int nids, int *S, int *V) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long long fsz = ftell(f);
    /* --eval-ids 默认在流首插 BOS(行数=nids+1), --eval-no-bos 则不插(行数=nids)。
     * 两种都要能读: 按行数反推 VOCAB, 只有一种能整除(129280 词表下不会撞车)。 */
    long long v = 0, skip = 0;
    if (nids > 0 && fsz % ((long long)(nids + 1) * 4) == 0) {
        v = fsz / ((long long)(nids + 1) * 4); skip = 1;          /* 首行是 BOS 的输出, 掐掉 */
    } else if (nids > 0 && fsz % ((long long)nids * 4) == 0) {
        v = fsz / ((long long)nids * 4); skip = 0;
    } else {
        fprintf(stderr, "%s: %lld 字节 与 ids 数 %d 对不上(含/不含 BOS 都除不尽)\n", path, fsz, nids);
        exit(2);
    }
    if (v < 1024 || v > (1 << 22)) { fprintf(stderr, "%s: 反推 VOCAB=%lld 不合理\n", path, v); exit(2); }
    fprintf(stderr, "%s: %d 行 × VOCAB %lld%s\n", path, nids, v, skip ? " (掐掉流首 BOS 行)" : " (无 BOS)");
    fseek(f, (long)(skip * v * 4), SEEK_SET);
    float *lg = malloc((size_t)nids * (size_t)v * sizeof(float));
    if (!lg || fread(lg, 4, (size_t)nids * (size_t)v, f) != (size_t)nids * (size_t)v) {
        fprintf(stderr, "%s: logits 截断\n", path); exit(2);
    }
    fclose(f);
    *S = nids; *V = (int)v;
    return lg;
}

static float *read_student(const char *path, int *S, int *V) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    int32_t sv[2];
    if (fread(sv, 4, 2, f) != 2) { fprintf(stderr, "%s: 头截断\n", path); exit(2); }
    *S = sv[0]; *V = sv[1];
    size_t n = (size_t)*S * *V;
    float *lg = malloc(n * sizeof(float));
    if (!lg || fread(lg, 4, n, f) != n) { fprintf(stderr, "%s: 学生 logits 截断\n", path); exit(2); }
    fclose(f);
    return lg;
}

/* 行内 log-softmax → out(f64); 返回 logsumexp 供复用 */
static void log_softmax_row(const float *x, double *out, int V) {
    float mx = x[0];
    for (int v = 1; v < V; v++) if (x[v] > mx) mx = x[v];
    double s = 0;
    for (int v = 0; v < V; v++) s += exp((double)x[v] - mx);
    double lse = log(s);
    for (int v = 0; v < V; v++) out[v] = (double)x[v] - mx - lse;
}

/* teacher-forced NLL: 只需要目标列 → 免整行展开 */
static double nll_at(const float *row, int V, int tgt) {
    float mx = row[0];
    for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
    double s = 0;
    for (int v = 0; v < V; v++) s += exp((double)row[v] - mx);
    return -((double)row[tgt] - mx - log(s));
}

/* top-K 索引(无序), 简单选择: 维护 K 槽最小值替换 */
static void topk_idx(const double *p, int V, int K, int *idx) {
    double vals[64]; int n = 0;
    for (int v = 0; v < V; v++) {
        if (n < K) { idx[n] = v; vals[n] = p[v]; n++; continue; }
        int mi = 0;
        for (int i = 1; i < K; i++) if (vals[i] < vals[mi]) mi = i;
        if (p[v] > vals[mi]) { vals[mi] = p[v]; idx[mi] = v; }
    }
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static int cmp_dbl_desc(const void *a, const void *b) {
    double d = *(const double *)b - *(const double *)a;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* ---- 逐位置指标 worker(位置块并行) ---- */
typedef struct {
    const float *ref, *stu;
    int V;
    const int *row;             /* [n] 绝对行号: 判决行集合可以不连续(见 --rows) */
    double *kld, *rms, *dtop, *smin;   /* [n] */
    uint8_t *same;              /* [n] */
} met_ctx;
static void met_worker(void *vc, int t0, int t1) {
    met_ctx *c = (met_ctx *)vc;
    int V = c->V;
    double *lr = malloc(V * sizeof(double)), *ls = malloc(V * sizeof(double));
    double *pr = malloc(V * sizeof(double)), *ps = malloc(V * sizeof(double));
    for (int t = t0; t < t1; t++) {
        const float *r = c->ref + (size_t)c->row[t] * V;
        const float *s = c->stu + (size_t)c->row[t] * V;
        log_softmax_row(r, lr, V); log_softmax_row(s, ls, V);
        double kl = 0; int ir = 0, iu = 0;
        for (int v = 0; v < V; v++) {
            pr[v] = exp(lr[v]); ps[v] = exp(ls[v]);
            kl += pr[v] * (lr[v] - ls[v]);
            if (r[v] > r[ir]) ir = v;
            if (s[v] > s[iu]) iu = v;
        }
        c->kld[t] = kl;
        { double sm = 0; for (int v = 0; v < V; v++) sm += pr[v] < ps[v] ? pr[v] : ps[v];
          c->smin[t] = sm; }   /* 分布还原率 Σmin(还原率铁律的主尺) */
        c->same[t] = (ir == iu);
        c->dtop[t] = fabs(pr[ir] - ps[ir]);
        /* top32 联合窗均方 */
        int ia[32], ib[32], u[64], nu = 0;
        topk_idx(pr, V, 32, ia); topk_idx(ps, V, 32, ib);
        memcpy(u, ia, sizeof ia); memcpy(u + 32, ib, sizeof ib);
        qsort(u, 64, sizeof(int), cmp_int);
        double d2 = 0; int cnt = 0;
        for (int i = 0; i < 64; i++) {
            if (i && u[i] == u[i - 1]) continue;
            double d = pr[u[i]] - ps[u[i]];
            d2 += d * d; cnt++; nu++;
        }
        (void)nu;
        c->rms[t] = d2 / cnt;
    }
    free(lr); free(ls); free(pr); free(ps);
}

typedef void (*pf_fn)(void *, int, int);
typedef struct { pf_fn fn; void *ctx; int i0, i1; } pf_arg;
static void *pf_tramp(void *a) { pf_arg *p = (pf_arg *)a; p->fn(p->ctx, p->i0, p->i1); return NULL; }
static void parallel_for(int n, int threads, pf_fn fn, void *ctx) {
    if (threads > n) threads = n > 0 ? n : 1;
    if (threads > 64) threads = 64;   /* 帽=th[64]/pa[64] 栈数组; 只影响速度不影响数值 */
    pthread_t th[64]; pf_arg pa[64];
    int per = (n + threads - 1) / threads, nt = 0;
    for (int t = 0; t < threads; t++) {
        int i0 = t * per, i1 = i0 + per > n ? n : i0 + per;
        if (i0 >= i1) break;
        pa[nt] = (pf_arg){fn, ctx, i0, i1};
        if (pthread_create(&th[nt], NULL, pf_tramp, &pa[nt])) { pa[nt].fn(ctx, i0, i1); continue; }
        nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

static double quantile(double *v, int n, double q) {   /* numpy 线性插值同义 */
    double *c = malloc(n * sizeof(double));
    memcpy(c, v, n * sizeof(double));
    /* 简单排序 */
    qsort(c, n, sizeof(double), (int (*)(const void *, const void *))cmp_dbl_desc);
    /* cmp_dbl_desc 是降序; 反转索引 */
    double pos = q * (n - 1);
    int i = (int)pos; double fr = pos - i;
    int ai = n - 1 - i, bi = ai - 1 < 0 ? 0 : ai - 1;   /* 升序位置映射 */
    double lo_v = c[ai], hi_v = c[bi];
    free(c);
    return lo_v + (hi_v - lo_v) * fr;
}

typedef struct { double ppl, nll; int n; } pplr;
/* 行集合口径: row[] 是绝对行号(可不连续), 位置 i 预测 ids[i+1]。
 * nll = 目标 token 的平均负对数似然 —— ★后训练唯一的尺★(PPL=exp(nll) 只是它的另一种写法)。 */
static pplr ppl_block(const float *lg, const long *ids, int nids, int V, const int *row, int n) {
    pplr r = {NAN, NAN, 0};
    double s = 0; int used = 0;
    for (int t = 0; t < n; t++) {
        int i = row[t];
        if (i + 1 >= nids) continue;                     /* 末位没有下一个 token 可预测 */
        s += nll_at(lg + (size_t)i * V, V, (int)ids[i + 1]);
        used++;
    }
    if (!used) return r;
    r.nll = s / used; r.ppl = exp(r.nll); r.n = used;
    return r;
}

/* 引擎 --eval-nll 已经把每位置目标 token 的 NLL 算好了(f32[S]), 这里只按行取平均。
 * 口径与 ppl_block 逐式相同 —— 差别只在"谁来做那次 log_softmax": 那边现算, 这边引擎
 * 在同一趟前向里顺手算完。★为什么值得单开一条路★: 全词表 logits 每位置 517 KB,
 * 16 条样本要 46 GB, 统一内存机器上写它就是在掏 GPU 的内存(09-08 实撞崩机)。
 * 末位是 NaN(没有下一个 token 可预测), 跳过不计 —— 与 ppl_block 的 i+1>=nids 同义。 */
static pplr ppl_block_nll(const float *v, int S, const int *row, int n) {
    pplr r = {NAN, NAN, 0};
    double s = 0; int used = 0;
    for (int t = 0; t < n; t++) {
        int i = row[t];
        if (i < 0 || i >= S) continue;
        if (!(v[i] == v[i])) continue;                   /* NaN: 该位置没有目标 token */
        s += v[i]; used++;
    }
    if (!used) return r;
    r.nll = s / used; r.ppl = exp(r.nll); r.n = used;
    return r;
}

/* --rows "a:b" 或 "a:b,c:d,..." → 绝对行号数组(升序去重不做, 由调用方保证不重叠)。
 * 为什么要多段: 每条样本是[当日材料][本应写的报告], 只有报告那一段是后训练的目标 token,
 * 材料段不算数; N 条样本 = N 段不连续区间, 但要合成一个池子出一份平均 NLL。 */
static int *rows_parse(const char *spec, int S, int *out_n) {
    int cap = 64, n = 0;
    int *r = malloc((size_t)cap * sizeof(int));
    const char *p = spec;
    while (*p) {
        int a, b, adv = 0;
        if (sscanf(p, "%d:%d%n", &a, &b, &adv) != 2) { free(r); return NULL; }
        if (b > S) b = S;
        for (int i = a; i < b; i++) {
            if (i < 0) { free(r); return NULL; }
            if (n == cap) { cap *= 2; r = realloc(r, (size_t)cap * sizeof(int)); }
            r[n++] = i;
        }
        p += adv;
        if (*p == ',') p++;
        else if (*p) { free(r); return NULL; }
    }
    if (!n) { free(r); return NULL; }
    *out_n = n;
    return r;
}
static int *rows_range(int lo, int hi) {
    int *r = malloc((size_t)(hi - lo) * sizeof(int));
    for (int i = lo; i < hi; i++) r[i - lo] = i;
    return r;
}

int main(int argc, char **argv) {
    const char *refp = NULL, *rawp = NULL, *idsp = NULL, *stup = NULL, *sraw = NULL, *revl = NULL;
    int fit = 0, tail = 0, threads = 16; const char *rowout = NULL;   /* 逐位置 KL/Σmin/same 落盘(尾部集中度诊断, 不动五指标) */
    /* ★行偏移/行段(2026-09-08 夜间 z 微调)★ --xshift S0 N: 学生 logits 比锚少 N 行(锚行 [S0,S0+N) 是教师独有
     * 的事后上下文, 部署侧没有), 锚行 r>=S0+N ↔ 学生行 r−N(与 row_layout.inc.c 的 xshift 同一映射);
     * 上下文行用锚自己回填(KLD=0), 所以有偏移时必须配 --rows a:b 只看正文行, 不然全段被回填行冲稀。 */
    int xs0 = -1, xn = 0; const char *rowspec = NULL, *rnll = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--xshift") && i + 2 < argc) { xs0 = atoi(argv[++i]); xn = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rowspec = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) refp = argv[++i];
        else if (!strcmp(argv[i], "--ref-raw") && i + 1 < argc) rawp = argv[++i];
        else if (!strcmp(argv[i], "--ref-eval") && i + 1 < argc) revl = argv[++i];
        else if (!strcmp(argv[i], "--ref-nll") && i + 1 < argc) rnll = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) idsp = argv[++i];
        else if (!strcmp(argv[i], "--student") && i + 1 < argc) stup = argv[++i];
        else if (!strcmp(argv[i], "--stu-raw") && i + 1 < argc) sraw = argv[++i];
        else if (!strcmp(argv[i], "--fit") && i + 1 < argc) fit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tail") && i + 1 < argc) tail = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--row-out") && i + 1 < argc) rowout = argv[++i];
    }
    if ((!refp && !rawp && !revl && !rnll) || !idsp) { fprintf(stderr, "需 --ref/--ref-raw/--ref-eval/--ref-nll 与 --ids\n"); return 2; }

    /* ids: 动态扩容 —— 一条 8000 字的报告就 5k token, 17 条样本 11 万 token,
     * 原来的定长 65536 栈数组会静默截断(判决行落到序列外) */
    long *ids = NULL; int nids = 0, icap = 0;
    { FILE *f = fopen(idsp, "r");
      if (!f) { fprintf(stderr, "%s 打不开\n", idsp); return 2; }
      long v;
      while (fscanf(f, "%ld", &v) == 1) {
          if (nids == icap) { icap = icap ? icap * 2 : 8192;
                              long *nb = realloc(ids, (size_t)icap * sizeof(long));
                              if (!nb) { fprintf(stderr, "ids 扩容失败\n"); return 2; }
                              ids = nb; }
          ids[nids++] = v;
      }
      fclose(f);
      if (!nids) { fprintf(stderr, "%s: 一个 id 都没读到\n", idsp); return 2; } }

    /* --ref-nll: 引擎 --eval-nll 出的 f32[S], 只出 PPL/NLL 就退出。
     * 其余四指标(KLD/RMSΔp/Same top/Δp)要的是完整分布, 这条路上没有 —— 不糊弄, 直接不出。
     * 后训练判决要的本来也只有 NLL 这一个数(ppl_block 的注释: ★后训练唯一的尺★)。 */
    if (rnll) {
        FILE *f = fopen(rnll, "rb");
        if (!f) { fprintf(stderr, "%s 打不开\n", rnll); return 2; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz <= 0 || sz % 4) { fprintf(stderr, "%s: 不是 f32[S](%ld 字节)\n", rnll, sz); fclose(f); return 2; }
        int S = (int)(sz / 4);
        float *v = malloc((size_t)S * sizeof(float));
        if (!v || fread(v, 4, (size_t)S, f) != (size_t)S) { fprintf(stderr, "%s 读失败\n", rnll); fclose(f); return 2; }
        fclose(f);
        /* ids 只用来核对长度: NLL 已经是算好的, 但行数对不上就说明喂错了文件, 那种
         * "安静的错数"比报错难查十倍(定长 ids 截断那次就是这么栽的)。 */
        if (S != nids)
            printf("★注意: nll 文件 %d 行 vs ids %d 行 —— 行号口径不同会判错行★\n", S, nids);
        int rn = S; int *rr = rows_range(0, S); const char *nm = "全段";
        if (rowspec) {
            free(rr); rr = rows_parse(rowspec, S, &rn);
            if (!rr) { fprintf(stderr, "--rows 要 a:b 或 a:b,c:d,...(得到 \"%s\", S=%d)\n", rowspec, S); return 2; }
            nm = "行段";
            printf("行段 %s → n=%d 行(不连续段已合成一个池子)\n", rowspec, rn);
        }
        pplr r = ppl_block_nll(v, S, rr, rn);
        printf("参考 PPL[%s n=%d] = %.4f   (平均 NLL %.4f)\n", nm, r.n, r.ppl, r.nll);
        free(rr); free(v); free(ids);
        return 0;
    }

    meta_t m; float *ref;
    if (revl) { int S, V; ref = read_eval_raw(revl, nids, &S, &V);
                memset(&m, 0, sizeof m); m.S = S; m.VOCAB = V; m.NL = m.HCM = m.NACT = -1; }
    else if (rawp) { int S, V; ref = read_student(rawp, &S, &V);
                memset(&m, 0, sizeof m); m.S = S; m.VOCAB = V; m.NL = m.HCM = m.NACT = -1; }
    else ref = read_anchor(refp, &m);
    int S = m.S, V = m.VOCAB;
    if (nids > S) nids = S;
    printf("锚: S=%d VOCAB=%d NL=%d HCM=%d NACT=%d\n", S, V, m.NL, m.HCM, m.NACT);

    struct { const char *nm; int *row; int n; } segs[2] = {{"全段", NULL, 0}, {NULL, NULL, 0}};
    int nseg = 1;
    segs[0].row = rows_range(0, S); segs[0].n = S;
    if (fit && fit < S) {
        free(segs[0].row);
        segs[0].nm = "held"; segs[0].row = rows_range(fit, S); segs[0].n = S - fit;
        segs[1].nm = "全段"; segs[1].row = rows_range(0, S);   segs[1].n = S; nseg = 2;
    }
    if (rowspec) {                                       /* --rows: 只判这些行(锚行号), 其余不出数 */
        for (int si = 0; si < nseg; si++) free(segs[si].row);
        int rn = 0; int *rr = rows_parse(rowspec, S, &rn);
        if (!rr) { fprintf(stderr, "--rows 要 a:b 或 a:b,c:d,...(得到 \"%s\", S=%d)\n", rowspec, S); return 2; }
        segs[0].nm = "行段"; segs[0].row = rr; segs[0].n = rn; nseg = 1;
        printf("行段 %s → n=%d 行(不连续段已合成一个池子)\n", rowspec, rn);
    }

    for (int si = 0; si < nseg; si++) {
        pplr r = ppl_block(ref, ids, nids, V, segs[si].row, segs[si].n);
        printf("参考 PPL[%s n=%d] = %.4f   (平均 NLL %.4f)\n", segs[si].nm, r.n, r.ppl, r.nll);
    }
    { long hit = 0, n = 0;
      for (int i = 0; i < S - 1 && i + 1 < nids; i++) {
          const float *row = ref + (size_t)i * V;
          int am = 0; for (int v = 1; v < V; v++) if (row[v] > row[am]) am = v;
          hit += (am == (int)ids[i + 1]); n++;
      }
      printf("参考 next-token top-1 命中 = %.1f%%   (FP 模型对本语料的自然可预测度)\n", 100.0 * hit / n); }

    if (!stup && !sraw) {
        pplr r = ppl_block(ref, ids, nids, V, segs[0].row, segs[0].n);
        int ok = isfinite(r.ppl) && r.ppl > 1.0 && r.ppl < 30.0;
        printf(ok ? "★冒烟判决: PASS(前向/读权重正确)★\n" : "★冒烟判决: FAIL(PPL=%f)★\n", r.ppl);
        return ok ? 0 : 1;
    }

    int Ss, Vs; float *stu;
    if (sraw) {
        /* 裸 f32 [n][V] 无头(--eval-logits 批量路输出); 首行=BOS 输出, 掐掉后取前 S 行 */
        FILE *f = fopen(sraw, "rb");
        if (!f) { fprintf(stderr, "%s 打不开\n", sraw); return 2; }
        fseek(f, 0, SEEK_END); long long fsz = ftell(f);
        long long rows = fsz / ((long long)V * 4);
        if (rows < S + 1) { fprintf(stderr, "%s 行不足 %lld < %d+BOS\n", sraw, rows, S); return 2; }
        fseek(f, (long)((long long)V * 4), SEEK_SET);
        stu = malloc((size_t)S * V * sizeof(float));
        if (!stu || fread(stu, 4, (size_t)S * V, f) != (size_t)S * V) { fprintf(stderr, "%s 读断\n", sraw); return 2; }
        fclose(f);
        Ss = S; Vs = V;
    } else {
        stu = read_student(stup, &Ss, &Vs);
    }
    if (xn > 0) {                                        /* 学生行 → 锚行(映射见 --xshift 注释) */
        if (xs0 < 0 || xs0 + xn > S) { fprintf(stderr, "--xshift %d %d 越界(S=%d)\n", xs0, xn, S); return 2; }
        if (Ss != S - xn || Vs != V) { fprintf(stderr, "形状不齐(偏移口径): ref(%d,%d) 期望学生 (%d,%d) 实得 (%d,%d)\n", S, V, S - xn, V, Ss, Vs); return 2; }
        float *full = malloc((size_t)S * V * sizeof(float));
        if (!full) { fprintf(stderr, "内存\n"); return 2; }
        for (int r = 0; r < S; r++) {
            int src = r < xs0 ? r : (r >= xs0 + xn ? r - xn : -1);
            memcpy(full + (size_t)r * V, src < 0 ? ref + (size_t)r * V : stu + (size_t)src * V, (size_t)V * sizeof(float));
        }
        free(stu); stu = full; Ss = S;
        printf("行偏移: 锚行 [%d,%d) 教师独有上下文(用锚回填, 不算数), 锚行 >= %d ↔ 学生行 − %d%s\n",
               xs0, xs0 + xn, xs0 + xn, xn, rowspec ? "" : " ★未给 --rows, 全段含回填行★");
    }
    if (Ss != S || Vs != V) { fprintf(stderr, "形状不齐: ref(%d,%d) stu(%d,%d)\n", S, V, Ss, Vs); return 2; }

    for (int si = 0; si < nseg; si++) {
        const int *row = segs[si].row; int n = segs[si].n;
        double *kld = malloc(n * sizeof(double)), *rms = malloc(n * sizeof(double));
        double *dtop = malloc(n * sizeof(double)), *smin = malloc(n * sizeof(double));
        uint8_t *same = malloc(n);
        met_ctx mc = {ref, stu, V, row, kld, rms, dtop, smin, same};
        parallel_for(n, threads, met_worker, &mc);
        pplr sp = ppl_block(stu, ids, nids, V, row, n);
        pplr rp = ppl_block(ref, ids, nids, V, row, n);
        double mk = 0, mr = 0, ms = 0, md = 0, sm = 0;
        for (int t = 0; t < n; t++) { mk += kld[t]; mr += rms[t]; ms += same[t]; md += dtop[t]; sm += smin[t]; }
        printf("\n== %s (n=%d) ==\n", segs[si].nm, n);
        printf("  PPL(student)    = %.4f   (ref %.4f, 比值 %.3f)\n", sp.ppl, rp.ppl, sp.ppl / rp.ppl);
        printf("  平均 NLL(学生)  = %.4f   (ref %.4f, 差 %+.4f) ★后训练的尺: 越低越好★\n",
               sp.nll, rp.nll, sp.nll - rp.nll);
        printf("  分布还原率 Σmin = %.4f   (中位 %.4f, p5 %.4f) ★主尺, 对标≥0.90★\n",
               sm / n, quantile(smin, n, 0.5), quantile(smin, n, 0.05));
        printf("  Mean KLD        = %.5f   (中位 %.5f, p95 %.5f)\n", mk / n,
               quantile(kld, n, 0.5), quantile(kld, n, 0.95));
        printf("  RMS Δp(top32窗) = %.4f%%\n", sqrt(mr / n) * 100);
        printf("  Same top token  = %.2f%%\n", ms / n * 100);
        printf("  Δp(ref top tok) = %.4f (p95 %.4f)\n", md / n, quantile(dtop, n, 0.95));
        if (rowout) {   /* 行=位置 kld smin same(与上面五指标同一批数, 只是不聚合) */
            FILE *fo = fopen(rowout, si == 0 ? "w" : "a");
            if (fo) { for (int t = 0; t < n; t++) fprintf(fo, "%d %.6f %.6f %d\n", row[t], kld[t], smin[t], same[t]); fclose(fo); }
        }
        free(kld); free(rms); free(dtop); free(smin); free(same);
    }

    if (tail) {
        /* 判决行集合与上面同一个(segs[0]) —— 给了 --rows 就只查那些行, 否则 held/全段。 */
        const int *trow = segs[0].row; int n = 0;
        int *tr = malloc((size_t)segs[0].n * sizeof(int));
        for (int t = 0; t < segs[0].n; t++) if (trow[t] + 1 < nids) tr[n++] = trow[t];
        double *d = malloc(n * sizeof(double)), *nr = malloc(n * sizeof(double)), *ns = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            int tgt = (int)ids[tr[i] + 1];
            nr[i] = nll_at(ref + (size_t)tr[i] * V, V, tgt);
            ns[i] = nll_at(stu + (size_t)tr[i] * V, V, tgt);
            d[i] = ns[i] - nr[i];
        }
        double gap = 0; for (int i = 0; i < n; i++) gap += d[i];
        /* top-N 索引(降序) */
        int *ord = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) ord[i] = i;
        for (int a = 0; a < tail && a < n; a++) {     /* 部分选择排序, tail 小 */
            int mi = a;
            for (int b = a + 1; b < n; b++) if (d[ord[b]] > d[ord[mi]]) mi = b;
            int tmp = ord[a]; ord[a] = ord[mi]; ord[mi] = tmp;
        }
        double topsum = 0; for (int a = 0; a < tail && a < n; a++) topsum += d[ord[a]];
        printf("\n== 尾部报表(判决行 ΔNLL=stu−ref, 总log差=%.2f, 均值=%.4f) ==\n", gap, gap / n);
        printf("  top%d token 承担全部 PPL 差的 %.0f%%\n", tail, topsum / (gap > 1e-9 ? gap : 1e-9) * 100);
        for (int a = 0; a < tail && a < n; a++) {
            int i = ord[a], tgt = (int)ids[tr[i] + 1];
            const float *rr = ref + (size_t)tr[i] * V, *ss = stu + (size_t)tr[i] * V;
            int rt = 0, st = 0;
            for (int v = 1; v < V; v++) { if (rr[v] > rr[rt]) rt = v; if (ss[v] > ss[st]) st = v; }
            printf("  pos=%d tgt=%d dNLL=%+.3f (ref%.2f->stu%.2f) refTop=%d stuTop=%d%s\n",
                   tr[i], tgt, d[i], nr[i], ns[i], rt, st, ns[i] > 6 ? " ★" : " ");
        }
        free(d); free(nr); free(ns); free(ord); free(tr);
    }
    free(ref); free(stu);
    return 0;
}
