/* linalg_small.c — implementation of the tiny dense solver declared in
 * linalg_small.h. Double precision throughout. Pure C99.
 *
 * Scope: the per-layer four-loss closed-form solve (SPEC.md §7) only ever
 * forms small dense systems (a few hundred to a few thousand unknowns), so
 * the straightforward O(n^3) factorization and O(n^2)-per-step power
 * iteration here are entirely adequate; clarity is favored over blocking.
 */
#include "linalg_small.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* internal helpers (not part of the public API)                       */
/* ------------------------------------------------------------------ */

/* Minimal row-major double GEMM: C(m×q) = A(m×p) · B(p×q).
 * Used internally for the matrix-vector products of power iteration (q=1).
 * Tiny on purpose — not exposed in the header. */
static void dmm(const double *A, const double *B, double *C,
                int m, int p, int q) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < q; j++) {
            double s = 0.0;
            for (int t = 0; t < p; t++)
                s += A[(size_t)i * p + t] * B[(size_t)t * q + j];
            C[(size_t)i * q + j] = s;
        }
    }
}

/* splitmix64 — a fast, well-distributed 64-bit PRNG. We use it (and ONLY
 * it) for sym_eig_topk's start vectors so results are reproducible from the
 * seed alone. Note: state==0 still yields a good stream (the first add
 * perturbs it), so seed 0 needs no special handling. */
static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* In-place L2-normalize; returns the (pre-normalization) norm. If the norm
 * is 0 the vector is left untouched and 0 is returned (caller guards). */
static double normalize_vec(double *v, int n) {
    double s = 0.0;
    for (int t = 0; t < n; t++) s += v[t] * v[t];
    s = sqrt(s);
    if (s > 0.0) {
        double inv = 1.0 / s;
        for (int t = 0; t < n; t++) v[t] *= inv;
    }
    return s;
}

/* Modified Gram-Schmidt: subtract from v its projection onto each of the m
 * rows of `basis` (n-length each). The rows are assumed already unit-norm
 * (the eigenvectors we deflate against are normalized when stored), so the
 * projection coefficient is just the dot product. */
static void orth_against(double *v, const double *basis, int m, int n) {
    for (int j = 0; j < m; j++) {
        const double *bj = basis + (size_t)j * n;
        double d = 0.0;
        for (int t = 0; t < n; t++) d += v[t] * bj[t];
        for (int t = 0; t < n; t++) v[t] -= d * bj[t];
    }
}

/* ------------------------------------------------------------------ */
/* public: SPD solve via Cholesky                                      */
/* ------------------------------------------------------------------ */

int chol_solve_spd(double *A, int n, double *B, int nrhs) {
    if (n <= 0 || nrhs <= 0 || !A || !B) return 0; /* nothing to do */

    /* Cholesky-Banachiewicz factorization, column by column, writing L into
     * the lower triangle of A. For column j we use only already-computed L
     * columns (kk < j) and the original lower-triangle entries of column j,
     * so reading and overwriting can share storage safely. The upper
     * triangle of A is never touched. */
    for (int j = 0; j < n; j++) {
        double sum = A[(size_t)j * n + j];            /* original diagonal */
        for (int kk = 0; kk < j; kk++) {
            double Ljk = A[(size_t)j * n + kk];
            sum -= Ljk * Ljk;
        }
        if (sum <= 0.0) return 1;                     /* not SPD: bad pivot */
        double Ljj = sqrt(sum);
        A[(size_t)j * n + j] = Ljj;
        double invLjj = 1.0 / Ljj;
        for (int i = j + 1; i < n; i++) {
            double s = A[(size_t)i * n + j];          /* original L_ij seed */
            for (int kk = 0; kk < j; kk++)
                s -= A[(size_t)i * n + kk] * A[(size_t)j * n + kk];
            A[(size_t)i * n + j] = s * invLjj;
        }
    }

    /* Solve A·x = b column by column: forward solve L·y = b, then back solve
     * Lᵀ·x = y. Lᵀ[i][kk] == L[kk][i] == A[kk*n + i]. */
    for (int c = 0; c < nrhs; c++) {
        /* forward substitution: L y = b */
        for (int i = 0; i < n; i++) {
            double s = B[(size_t)i * nrhs + c];
            for (int kk = 0; kk < i; kk++)
                s -= A[(size_t)i * n + kk] * B[(size_t)kk * nrhs + c];
            B[(size_t)i * nrhs + c] = s / A[(size_t)i * n + i];
        }
        /* back substitution: Lᵀ x = y */
        for (int i = n - 1; i >= 0; i--) {
            double s = B[(size_t)i * nrhs + c];
            for (int kk = i + 1; kk < n; kk++)
                s -= A[(size_t)kk * n + i] * B[(size_t)kk * nrhs + c];
            B[(size_t)i * nrhs + c] = s / A[(size_t)i * n + i];
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* public: top-k symmetric eigenpairs via power iteration + deflation  */
/* ------------------------------------------------------------------ */

void sym_eig_topk(const double *M, int n, int k,
                  double *evecs, double *evals, int iters, uint64_t seed) {
    if (n <= 0 || k <= 0 || !M || !evecs || !evals) return;
    if (k > n) k = n;                 /* at most n eigenpairs exist */
    if (iters <= 0) iters = 200;      /* default budget; geometric convergence */

    uint64_t st = seed;               /* deterministic PRNG state */
    double *w = (double *)malloc((size_t)n * sizeof(double));
    if (!w) return;

    for (int i = 0; i < k; i++) {
        double *vi = evecs + (size_t)i * n;

        /* deterministic random start in [-1,1)^n from splitmix64(seed) */
        for (int t = 0; t < n; t++) {
            uint64_t r = splitmix64(&st);
            /* top 53 bits -> uniform double in [0,1), then map to [-1,1) */
            double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
            vi[t] = 2.0 * u - 1.0;
        }
        /* deflate the start against eigenvectors already found, normalize */
        orth_against(vi, evecs, i, n);
        if (normalize_vec(vi, n) == 0.0) {
            /* start collapsed into the found subspace; retry from a
             * coordinate axis so we still emit a valid unit vector */
            for (int t = 0; t < n; t++) vi[t] = (t == (i % n)) ? 1.0 : 0.0;
            orth_against(vi, evecs, i, n);
            if (normalize_vec(vi, n) == 0.0) { evals[i] = 0.0; continue; }
        }

        /* Power iteration on the deflated operator. Re-orthogonalizing the
         * iterate against the found eigenvectors every step keeps the
         * deflation numerically clean (rounding otherwise lets the dominant
         * component creep back in). Converges to the largest-magnitude
         * eigenvector of M restricted to the orthogonal complement. */
        for (int it = 0; it < iters; it++) {
            dmm(M, vi, w, n, n, 1);              /* w = M·vi */
            orth_against(w, evecs, i, n);
            double nrm = normalize_vec(w, n);
            if (nrm == 0.0) break;               /* remaining spectrum exhausted */
            memcpy(vi, w, (size_t)n * sizeof(double));
        }

        /* Rayleigh quotient λ = viᵀ·(M·vi) is the eigenvalue for unit vi.
         * (It converges quadratically faster than the eigenvector.) */
        dmm(M, vi, w, n, n, 1);
        double lam = 0.0;
        for (int t = 0; t < n; t++) lam += vi[t] * w[t];
        evals[i] = lam;
    }
    free(w);

    /* Enforce the descending-eigenvalue postcondition. k is tiny, so an
     * O(k^2) selection sort (swapping eigenvector rows in lockstep) is fine. */
    for (int a = 0; a < k; a++) {
        int best = a;
        for (int b = a + 1; b < k; b++)
            if (evals[b] > evals[best]) best = b;
        if (best != a) {
            double tv = evals[a]; evals[a] = evals[best]; evals[best] = tv;
            double *ra = evecs + (size_t)a * n;
            double *rb = evecs + (size_t)best * n;
            for (int t = 0; t < n; t++) {
                double tmp = ra[t]; ra[t] = rb[t]; rb[t] = tmp;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* self-test:  cc -std=c99 -O2 -DLINALG_TEST linalg_small.c -o a2test -lm */
/* ------------------------------------------------------------------ */
#ifdef LINALG_TEST
#include <stdio.h>

/* uniform double in [-1,1) from the same generator, for building test data */
static double rnd_pm1(uint64_t *s) {
    uint64_t r = splitmix64(s);
    return 2.0 * ((double)(r >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
}

/* (1) SPD solve: A = MᵀM + I (guaranteed positive definite), pick a known
 *     X, set B = A·X, solve, check ‖X̂−X‖∞. Exercises nrhs>1 and the
 *     not-positive-definite return path. */
static int test_chol(void) {
    const int n = 6, nrhs = 3;
    uint64_t s = 0xC0FFEEULL;

    double *Mrand = (double *)malloc((size_t)n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) Mrand[i] = rnd_pm1(&s);

    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int t = 0; t < n; t++)
                acc += Mrand[(size_t)t * n + i] * Mrand[(size_t)t * n + j];
            A[(size_t)i * n + j] = acc + (i == j ? 1.0 : 0.0);
        }

    double *X = (double *)malloc((size_t)n * nrhs * sizeof(double));
    for (int i = 0; i < n * nrhs; i++) X[i] = rnd_pm1(&s);

    double *B = (double *)malloc((size_t)n * nrhs * sizeof(double));
    dmm(A, X, B, n, n, nrhs);                 /* B = A·X */

    int rc = chol_solve_spd(A, n, B, nrhs);   /* A→L, B→X̂ */

    double maxerr = 0.0;
    for (int i = 0; i < n * nrhs; i++) {
        double e = fabs(B[i] - X[i]);
        if (e > maxerr) maxerr = e;
    }
    printf("[chol] rc=%d  nrhs=%d  max|x_hat - x| = %.3e\n", rc, nrhs, maxerr);

    /* not-PD detection: -I has a negative pivot immediately */
    double *N = (double *)malloc((size_t)n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) N[i] = 0.0;
    for (int i = 0; i < n; i++) N[(size_t)i * n + i] = -1.0;
    double dummy[6] = {1, 1, 1, 1, 1, 1};
    int rc2 = chol_solve_spd(N, n, dummy, 1);
    printf("[chol] non-PD (-I) -> rc=%d (expect nonzero)\n", rc2);

    int ok = (rc == 0) && (maxerr < 1e-8) && (rc2 != 0);
    free(Mrand); free(A); free(X); free(B); free(N);
    return ok;
}

/* (2) Eigen: A = λ1 v1 v1ᵀ + λ2 v2 v2ᵀ with orthonormal v1,v2 → known top-2
 *     eigenpairs. Check eigenvalues, |cosine| of recovered eigenvectors,
 *     and bit-identical determinism across two equally-seeded runs. */
static int test_eig(void) {
    const int n = 8, k = 2;
    uint64_t s = 0x1234ULL;

    double v1[8], v2[8];
    for (int i = 0; i < n; i++) v1[i] = rnd_pm1(&s);
    normalize_vec(v1, n);
    for (int i = 0; i < n; i++) v2[i] = rnd_pm1(&s);
    {   /* orthogonalize v2 against v1, then normalize */
        double d = 0.0;
        for (int i = 0; i < n; i++) d += v2[i] * v1[i];
        for (int i = 0; i < n; i++) v2[i] -= d * v1[i];
    }
    normalize_vec(v2, n);

    const double lam1 = 10.0, lam2 = 3.0;
    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[(size_t)i * n + j] = lam1 * v1[i] * v1[j] + lam2 * v2[i] * v2[j];

    double evecs[2 * 8], evals[2];
    double evecs2[2 * 8], evals2[2];
    sym_eig_topk(A, n, k, evecs,  evals,  300, 0xABCDEF01ULL);
    sym_eig_topk(A, n, k, evecs2, evals2, 300, 0xABCDEF01ULL); /* same seed */

    double c0 = 0.0, c1 = 0.0;
    for (int i = 0; i < n; i++) {
        c0 += evecs[i]     * v1[i];
        c1 += evecs[n + i] * v2[i];
    }
    c0 = fabs(c0); c1 = fabs(c1);

    double e_lam1 = fabs(evals[0] - lam1);
    double e_lam2 = fabs(evals[1] - lam2);
    double e_c0 = fabs(1.0 - c0);
    double e_c1 = fabs(1.0 - c1);
    printf("[eig]  evals = [%.6f, %.6f]  (expect %.1f, %.1f)\n",
           evals[0], evals[1], lam1, lam2);
    printf("[eig]  |cos(v1)|=%.10f  |cos(v2)|=%.10f\n", c0, c1);
    printf("[eig]  err: dλ1=%.3e dλ2=%.3e d|cos1|=%.3e d|cos2|=%.3e\n",
           e_lam1, e_lam2, e_c0, e_c1);

    int det = (memcmp(evals, evals2, sizeof(evals)) == 0) &&
              (memcmp(evecs, evecs2, sizeof(evecs)) == 0);
    printf("[eig]  deterministic across two runs: %s\n",
           det ? "YES (bit-identical)" : "NO");

    int ok = (e_lam1 < 1e-4) && (e_lam2 < 1e-4) &&
             (e_c0 < 1e-4) && (e_c1 < 1e-4) && det;
    free(A);
    return ok;
}

int main(void) {
    int a = test_chol();
    int b = test_eig();
    printf("\nRESULT: chol=%s  eig=%s  =>  %s\n",
           a ? "PASS" : "FAIL", b ? "PASS" : "FAIL",
           (a && b) ? "ALL PASS" : "FAIL");
    return (a && b) ? 0 : 1;
}
#endif /* LINALG_TEST */
