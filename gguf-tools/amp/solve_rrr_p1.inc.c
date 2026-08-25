/* solve_rrr.c — aggregate-level four-loss closed-form solve (see solve_rrr.h).
 *
 * Shapes in production: d=4096, n_exp=256, topk=6, n_train ≈ a few thousand
 * Go tokens. Heavy pieces (d×d Gram, n×d×d fitted-value matmul, d×d fitted
 * covariance) are threaded over row blocks; everything else is O(n·d·k) or
 * O(k·E²) and stays single-threaded. All accumulation in double.
 */
#include "solve_rrr.h"
#include "linalg_small.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void rrr_params_default(rrr_params *p) {
    memset(p, 0, sizeof *p);
    p->k = 64;
    p->lambda_rel = 1e-4;
    p->lam_c_rel = 1e-2;
    p->w_smooth = 0.0;
    p->w_align = 0.0;
    p->chunk = 512;
    p->alpha_router = 0.0;
    p->Wr = NULL;
    p->procrustes = 1;
    p->seed = 1234567891234567ULL;
    p->eig_iters = 0;
    p->threads = 6;
}

/* ------------------------------------------------------------------ */
/* tiny deterministic RNG (matches the repo's no-rand() rule)           */
/* ------------------------------------------------------------------ */
#ifdef RRR_TEST
static uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
#endif

/* ------------------------------------------------------------------ */
/* parallel_for over row blocks                                        */
/* ------------------------------------------------------------------ */
typedef void (*pf_fn)(void *ctx, int i0, int i1);
typedef struct { pf_fn fn; void *ctx; int i0, i1; } pf_arg;
static void *pf_tramp(void *a) { pf_arg *p = (pf_arg *)a; p->fn(p->ctx, p->i0, p->i1); return NULL; }
static void parallel_for(int n, int threads, pf_fn fn, void *ctx) {
    if (threads < 1) threads = 1;
    if (threads > n) threads = n > 0 ? n : 1;
    if (threads == 1) { fn(ctx, 0, n); return; }
    pthread_t th[64]; pf_arg args[64];
    if (threads > 64) threads = 64;
    int per = (n + threads - 1) / threads, nt = 0;
    for (int i0 = 0; i0 < n; i0 += per) {
        args[nt].fn = fn; args[nt].ctx = ctx; args[nt].i0 = i0;
        args[nt].i1 = i0 + per < n ? i0 + per : n;
        if (pthread_create(&th[nt], NULL, pf_tramp, &args[nt]) != 0) { fn(&args[nt], args[nt].i0, args[nt].i1); continue; }
        nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

/* ------------------------------------------------------------------ */
/* worker 1: lower-triangular Gram G=XᵀX and full RHS R=XᵀT            */
/* ------------------------------------------------------------------ */
typedef struct {
    const float *X, *T;   /* [n×d] each */
    int n, d;
    double *G;            /* [d×d] lower triangle filled                */
    double *R;            /* [d×d] full                                 */
} gram_ctx;
static void gram_worker(void *vc, int i0, int i1) {
    gram_ctx *c = (gram_ctx *)vc;
    const int d = c->d;
    for (int t = 0; t < c->n; t++) {
        const float *x = c->X + (size_t)t * d;
        const float *y = c->T + (size_t)t * d;
        for (int i = i0; i < i1; i++) {
            double xi = (double)x[i];
            if (xi == 0.0) continue;
            double *g = c->G + (size_t)i * d;
            double *r = c->R + (size_t)i * d;
            for (int j = 0; j <= i; j++) g[j] += xi * (double)x[j];
            for (int j = 0; j < d; j++)  r[j] += xi * (double)y[j];
        }
    }
}

/* threaded triangular substitution: after chol_solve_spd(A,n,B,0) factored A
 * in place (lower triangle = L), solve A·X = B for all nrhs columns in
 * parallel. Columns are independent → bit-identical to the serial path. B is
 * transposed to [nrhs×n] so each worker row is contiguous, then transposed
 * back. This is the dominant d²·nrhs phase of the per-layer solve. */
typedef struct { const double *Lf; double *Bt; int n; } trisolve_ctx;
static void trisolve_worker(void *vc, int c0, int c1) {
    trisolve_ctx *c = (trisolve_ctx *)vc;
    const int n = c->n;
    const double *L = c->Lf;
    for (int col = c0; col < c1; col++) {
        double *b = c->Bt + (size_t)col * n;
        for (int i = 0; i < n; i++) {              /* forward: L y = b */
            const double *Li = L + (size_t)i * n;
            double s = b[i];
            for (int j = 0; j < i; j++) s -= Li[j] * b[j];
            b[i] = s / Li[i];
        }
        for (int i = n - 1; i >= 0; i--) {         /* back: Lᵀ x = y */
            double s = b[i];
            for (int j = i + 1; j < n; j++) s -= L[(size_t)j * n + i] * b[j];
            b[i] = s / L[(size_t)i * n + i];
        }
    }
}
/* factor A in place, then threaded solve of A·X=B (B row-major [n×nrhs],
 * overwritten). chol_solve_spd short-circuits on nrhs=0, so factorization is
 * triggered with one throwaway RHS column (d² flops, negligible). */
static int chol_solve_mt(double *A, int n, double *B, int nrhs, int threads) {
    double *dummy = (double *)calloc((size_t)n, sizeof(double));
    if (!dummy) return 1;
    int frc = chol_solve_spd(A, n, dummy, 1);           /* factor + trivial solve */
    free(dummy);
    if (frc != 0) return 1;
    double *Bt = (double *)malloc((size_t)n * nrhs * sizeof(double));
    if (!Bt) return 1;
    for (int i = 0; i < n; i++)
        for (int c = 0; c < nrhs; c++) Bt[(size_t)c * n + i] = B[(size_t)i * nrhs + c];
    trisolve_ctx tc = { A, Bt, n };
    parallel_for(nrhs, threads, trisolve_worker, &tc);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < nrhs; c++) B[(size_t)i * nrhs + c] = Bt[(size_t)c * n + i];
    free(Bt);
    return 0;
}

/* worker 2: fitted values Yfit = X·Mᵀ  (MT is [d_in×d_out] row-major) */
typedef struct {
    const float *X; const double *MT; float *Yfit; int n, d;
} fit_ctx;
static void fit_worker(void *vc, int t0, int t1) {
    fit_ctx *c = (fit_ctx *)vc;
    const int d = c->d;
    double *acc = (double *)malloc((size_t)d * sizeof(double));
    if (!acc) return;
    for (int t = t0; t < t1; t++) {
        const float *x = c->X + (size_t)t * d;
        memset(acc, 0, (size_t)d * sizeof(double));
        for (int i = 0; i < d; i++) {
            double xi = (double)x[i];
            if (xi == 0.0) continue;
            const double *m = c->MT + (size_t)i * d;
            for (int o = 0; o < d; o++) acc[o] += xi * m[o];
        }
        float *yf = c->Yfit + (size_t)t * d;
        for (int o = 0; o < d; o++) yf[o] = (float)acc[o];
    }
    free(acc);
}

/* worker 3: fitted covariance F = YfitᵀYfit (lower triangle) */
typedef struct { const float *Yfit; int n, d; double *F; } fcov_ctx;
static void fcov_worker(void *vc, int i0, int i1) {
    fcov_ctx *c = (fcov_ctx *)vc;
    const int d = c->d;
    for (int t = 0; t < c->n; t++) {
        const float *y = c->Yfit + (size_t)t * d;
        for (int i = i0; i < i1; i++) {
            double yi = (double)y[i];
            if (yi == 0.0) continue;
            double *f = c->F + (size_t)i * d;
            for (int j = 0; j <= i; j++) f[j] += yi * (double)y[j];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Q^{±1/2} appliers: Q = I + α²·WrᵀWr, rank-n_exp Woodbury correction  */
/* Q^{p} = I + Wrᵀ·P·diag(((1+α²μ_i)^p − 1)/μ_i)·Pᵀ·Wr, eig WrWrᵀ=PDPᵀ  */
/* ------------------------------------------------------------------ */
typedef struct {
    int d, ne, ok;
    const float *Wr;
    double *P;      /* [ne×ne] rows = eigenvectors of WrWrᵀ */
    double *mu;     /* [ne] eigenvalues */
} qmet;
static int qmet_build(qmet *q, const float *Wr, int ne, int d, uint64_t seed) {
    memset(q, 0, sizeof *q);
    q->d = d; q->ne = ne; q->Wr = Wr;
    double *S = (double *)calloc((size_t)ne * ne, sizeof(double));
    q->P = (double *)malloc((size_t)ne * ne * sizeof(double));
    q->mu = (double *)malloc((size_t)ne * sizeof(double));
    if (!S || !q->P || !q->mu) { free(S); return 1; }
    for (int a = 0; a < ne; a++) {
        const float *wa = Wr + (size_t)a * d;
        for (int b = 0; b <= a; b++) {
            const float *wb = Wr + (size_t)b * d;
            double s = 0; for (int j = 0; j < d; j++) s += (double)wa[j] * (double)wb[j];
            S[(size_t)a * ne + b] = s; S[(size_t)b * ne + a] = s;
        }
    }
    sym_eig_topk(S, ne, ne, q->P, q->mu, 0, seed);
    free(S);
    q->ok = 1;
    return 0;
}
static void qmet_free(qmet *q) { free(q->P); free(q->mu); }
/* y ← Q^{pow/2} y for pow=+1|-1 (in place, single vector) */
static void qmet_apply(const qmet *q, double alpha, int pow, double *y, double *tmp_ne) {
    if (!q->ok) return;
    const int d = q->d, ne = q->ne;
    /* c = Wr·y */
    for (int a = 0; a < ne; a++) {
        const float *wa = q->Wr + (size_t)a * d;
        double s = 0; for (int j = 0; j < d; j++) s += (double)wa[j] * y[j];
        tmp_ne[a] = s;
    }
    /* c ← P·diag(f)·Pᵀ·c */
    double pc[1024];
    for (int i = 0; i < ne; i++) {
        const double *pi = q->P + (size_t)i * ne;
        double s = 0; for (int a = 0; a < ne; a++) s += pi[a] * tmp_ne[a];
        double mu = q->mu[i] > 1e-12 ? q->mu[i] : 1e-12;
        double lam = 1.0 + alpha * alpha * mu;
        double f = (pow > 0 ? sqrt(lam) : 1.0 / sqrt(lam)) - 1.0;
        pc[i] = s * (f / mu);
    }
    for (int a = 0; a < ne; a++) {
        double s = 0;
        for (int i = 0; i < ne; i++) s += q->P[(size_t)i * ne + a] * pc[i];
        tmp_ne[a] = s;
    }
    /* y += Wrᵀ·c */
    for (int a = 0; a < ne; a++) {
        double ca = tmp_ne[a];
        if (ca == 0.0) continue;
        const float *wa = q->Wr + (size_t)a * d;
        for (int j = 0; j < d; j++) y[j] += ca * (double)wa[j];
    }
}

/* ------------------------------------------------------------------ */
/* fidelity: mean token cosine(pred, ref) + global rel-L2               */
/* ------------------------------------------------------------------ */
static void agg_metrics(const float *Yhat, const float *Yref, const double *corr /*may be NULL*/,
                        const int *idx, int n, int d, double *cos_out, double *rel_out) {
    double csum = 0, num = 0, den = 0; long cnt = 0;
    for (int s = 0; s < n; s++) {
        int t = idx ? idx[s] : s;
        const float *h = Yhat + (size_t)t * d;
        const float *r = Yref + (size_t)t * d;
        const double *co = corr ? corr + (size_t)s * d : NULL;
        double dot = 0, np = 0, nr = 0;
        for (int j = 0; j < d; j++) {
            double pj = (double)h[j] + (co ? co[j] : 0.0);
            double rj = (double)r[j];
            dot += pj * rj; np += pj * pj; nr += rj * rj;
            double e = rj - pj; num += e * e; den += rj * rj;
        }
        if (np > 0 && nr > 0) { csum += dot / (sqrt(np) * sqrt(nr)); cnt++; }
    }
    *cos_out = cnt ? csum / cnt : 0.0;
    *rel_out = den > 0 ? sqrt(num / den) : 0.0;
}

/* per-token latent v=V·x, expert-summed gains sC, correction U·(sC⊙v)+b */
static void token_corr(const double *U, const double *V, const double *C, const double *b,
                       const float *x, const float *route, int topk,
                       int d, int k, int n_exp, double *corr /*[d]*/, double *vtmp /*[k]*/) {
    for (int kk = 0; kk < k; kk++) {
        const double *vr = V + (size_t)kk * d;
        double s = 0; for (int j = 0; j < d; j++) s += vr[j] * (double)x[j];
        vtmp[kk] = s;
    }
    for (int j = 0; j < d; j++) corr[j] = b[j] * (double)topk; /* stored b = b_fit/topk */
    for (int kk = 0; kk < k; kk++) {
        double sc = 0;
        for (int e = 0; e < topk; e++) {
            int id = (int)route[e];
            if (id >= 0 && id < n_exp) sc += C[(size_t)id * k + kk];
        }
        double f = sc * vtmp[kk];
        if (f == 0.0) continue;
        for (int j = 0; j < d; j++) corr[j] += U[(size_t)j * k + kk] * f;
    }
}

/* ------------------------------------------------------------------ */
