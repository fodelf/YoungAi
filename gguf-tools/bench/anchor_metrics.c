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
 *       [--ref-raw raw.bin] [--fit N] [--tail N]                            */
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
    int V, lo;
    double *kld, *rms, *dtop, *smin;   /* [n] */
    uint8_t *same;              /* [n] */
} met_ctx;
static void met_worker(void *vc, int t0, int t1) {
    met_ctx *c = (met_ctx *)vc;
    int V = c->V;
    double *lr = malloc(V * sizeof(double)), *ls = malloc(V * sizeof(double));
    double *pr = malloc(V * sizeof(double)), *ps = malloc(V * sizeof(double));
    for (int t = t0; t < t1; t++) {
        const float *r = c->ref + (size_t)(c->lo + t) * V;
        const float *s = c->stu + (size_t)(c->lo + t) * V;
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

typedef struct { double ppl; int n; } pplr;
static pplr ppl_block(const float *lg, const long *ids, int nids, int V, int lo, int hi) {
    if (hi > nids - 1) hi = nids - 1;
    pplr r = {NAN, 0};
    if (hi <= lo) return r;
    double s = 0;
    for (int i = lo; i < hi; i++) s += nll_at(lg + (size_t)i * V, V, (int)ids[i + 1]);
    r.ppl = exp(s / (hi - lo)); r.n = hi - lo;
    return r;
}

int main(int argc, char **argv) {
    const char *refp = NULL, *rawp = NULL, *idsp = NULL, *stup = NULL, *sraw = NULL;
    int fit = 0, tail = 0, threads = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc) refp = argv[++i];
        else if (!strcmp(argv[i], "--ref-raw") && i + 1 < argc) rawp = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) idsp = argv[++i];
        else if (!strcmp(argv[i], "--student") && i + 1 < argc) stup = argv[++i];
        else if (!strcmp(argv[i], "--stu-raw") && i + 1 < argc) sraw = argv[++i];
        else if (!strcmp(argv[i], "--fit") && i + 1 < argc) fit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tail") && i + 1 < argc) tail = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    if ((!refp && !rawp) || !idsp) { fprintf(stderr, "需 --ref/--ref-raw 与 --ids\n"); return 2; }

    /* ids */
    long ids[65536]; int nids = 0;
    { FILE *f = fopen(idsp, "r");
      if (!f) { fprintf(stderr, "%s 打不开\n", idsp); return 2; }
      while (nids < 65536 && fscanf(f, "%ld", &ids[nids]) == 1) nids++;
      fclose(f); }

    meta_t m; float *ref;
    if (rawp) { int S, V; ref = read_student(rawp, &S, &V);
                memset(&m, 0, sizeof m); m.S = S; m.VOCAB = V; m.NL = m.HCM = m.NACT = -1; }
    else ref = read_anchor(refp, &m);
    int S = m.S, V = m.VOCAB;
    if (nids > S) nids = S;
    printf("锚: S=%d VOCAB=%d NL=%d HCM=%d NACT=%d\n", S, V, m.NL, m.HCM, m.NACT);

    struct { const char *nm; int lo, hi; } segs[2] = {{"全段", 0, S}, {NULL, 0, 0}};
    int nseg = 1;
    if (fit && fit < S) { segs[0].nm = "held"; segs[0].lo = fit; segs[1].nm = "全段"; segs[1].hi = S; nseg = 2; }

    for (int si = 0; si < nseg; si++) {
        pplr r = ppl_block(ref, ids, nids, V, segs[si].lo, segs[si].hi);
        printf("参考 PPL[%s n=%d] = %.4f\n", segs[si].nm, r.n, r.ppl);
    }
    { long hit = 0, n = 0;
      for (int i = 0; i < S - 1 && i + 1 < nids; i++) {
          const float *row = ref + (size_t)i * V;
          int am = 0; for (int v = 1; v < V; v++) if (row[v] > row[am]) am = v;
          hit += (am == (int)ids[i + 1]); n++;
      }
      printf("参考 next-token top-1 命中 = %.1f%%   (FP 模型对本语料的自然可预测度)\n", 100.0 * hit / n); }

    if (!stup && !sraw) {
        pplr r = ppl_block(ref, ids, nids, V, 0, S);
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
    if (Ss != S || Vs != V) { fprintf(stderr, "形状不齐: ref(%d,%d) stu(%d,%d)\n", S, V, Ss, Vs); return 2; }

    for (int si = 0; si < nseg; si++) {
        int lo = segs[si].lo, hi = segs[si].hi, n = hi - lo;
        double *kld = malloc(n * sizeof(double)), *rms = malloc(n * sizeof(double));
        double *dtop = malloc(n * sizeof(double)), *smin = malloc(n * sizeof(double));
        uint8_t *same = malloc(n);
        met_ctx mc = {ref, stu, V, lo, kld, rms, dtop, smin, same};
        parallel_for(n, threads, met_worker, &mc);
        pplr sp = ppl_block(stu, ids, nids, V, lo, hi);
        pplr rp = ppl_block(ref, ids, nids, V, lo, hi);
        double mk = 0, mr = 0, ms = 0, md = 0, sm = 0;
        for (int t = 0; t < n; t++) { mk += kld[t]; mr += rms[t]; ms += same[t]; md += dtop[t]; sm += smin[t]; }
        printf("\n== %s (n=%d) ==\n", segs[si].nm, n);
        printf("  PPL(student)    = %.4f   (ref %.4f, 比值 %.3f)\n", sp.ppl, rp.ppl, sp.ppl / rp.ppl);
        printf("  分布还原率 Σmin = %.4f   (中位 %.4f, p5 %.4f) ★主尺, 对标≥0.90★\n",
               sm / n, quantile(smin, n, 0.5), quantile(smin, n, 0.05));
        printf("  Mean KLD        = %.5f   (中位 %.5f, p95 %.5f)\n", mk / n,
               quantile(kld, n, 0.5), quantile(kld, n, 0.95));
        printf("  RMS Δp(top32窗) = %.4f%%\n", sqrt(mr / n) * 100);
        printf("  Same top token  = %.2f%%\n", ms / n * 100);
        printf("  Δp(ref top tok) = %.4f (p95 %.4f)\n", md / n, quantile(dtop, n, 0.95));
        free(kld); free(rms); free(dtop); free(smin); free(same);
    }

    if (tail) {
        int lo = (fit && fit < S) ? fit : 0, hi = S;
        if (hi > nids - 1) hi = nids - 1;
        int n = hi - lo;
        double *d = malloc(n * sizeof(double)), *nr = malloc(n * sizeof(double)), *ns = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            int tgt = (int)ids[lo + i + 1];
            nr[i] = nll_at(ref + (size_t)(lo + i) * V, V, tgt);
            ns[i] = nll_at(stu + (size_t)(lo + i) * V, V, tgt);
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
        printf("\n== 尾部报表(held ΔNLL=stu−ref, 总log差=%.2f, 均值=%.4f) ==\n", gap, gap / n);
        printf("  top%d token 承担全部 PPL 差的 %.0f%%\n", tail, topsum / (gap > 1e-9 ? gap : 1e-9) * 100);
        for (int a = 0; a < tail && a < n; a++) {
            int i = ord[a], tgt = (int)ids[lo + i + 1];
            const float *rr = ref + (size_t)(lo + i) * V, *ss = stu + (size_t)(lo + i) * V;
            int rt = 0, st = 0;
            for (int v = 1; v < V; v++) { if (rr[v] > rr[rt]) rt = v; if (ss[v] > ss[st]) st = v; }
            printf("  pos=%d tgt=%d dNLL=%+.3f (ref%.2f->stu%.2f) refTop=%d stuTop=%d%s\n",
                   lo + i, tgt, d[i], nr[i], ns[i], rt, st, ns[i] > 6 ? " ★" : " ");
        }
        free(d); free(nr); free(ns); free(ord);
    }
    free(ref); free(stu);
    return 0;
}
