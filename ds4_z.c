/* ds4_z.c -- z 隐变量: closed-form low-rank latent correction.
 * Self-contained (stdlib only); see ds4_z.h for the module contract. */
#include "ds4_z.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Deterministic LCG (Numerical Recipes constants): subspace-iteration init
 * must be reproducible, so no rand()/time seeding anywhere in this module. */
static inline uint32_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(*s >> 32);
}
static inline float lcg_unit(uint64_t *s) {
    return ((float)lcg_next(s) / 4294967296.0f) * 2.0f - 1.0f;   /* [-1,1) */
}

/* In-place Cholesky A = L L^T (A: d x d, lower triangle used). Returns 0, or
 * -1 if A is not positive definite (ridge lambda too small / degenerate X). */
static int cholesky(double *A, uint32_t d) {
    for (uint32_t j = 0; j < d; j++) {
        double diag = A[(size_t)j * d + j];
        for (uint32_t k = 0; k < j; k++) {
            double v = A[(size_t)j * d + k];
            diag -= v * v;
        }
        if (diag <= 0.0) return -1;
        diag = sqrt(diag);
        A[(size_t)j * d + j] = diag;
        for (uint32_t i = j + 1; i < d; i++) {
            double v = A[(size_t)i * d + j];
            for (uint32_t k = 0; k < j; k++) {
                v -= A[(size_t)i * d + k] * A[(size_t)j * d + k];
            }
            A[(size_t)i * d + j] = v / diag;
        }
    }
    return 0;
}

/* Solve L L^T w = b in place (b becomes w). */
static void cholesky_solve(const double *L, uint32_t d, double *b) {
    for (uint32_t i = 0; i < d; i++) {          /* forward: L y = b */
        double v = b[i];
        for (uint32_t k = 0; k < i; k++) v -= L[(size_t)i * d + k] * b[k];
        b[i] = v / L[(size_t)i * d + i];
    }
    for (uint32_t ii = d; ii-- > 0; ) {         /* backward: L^T w = y */
        double v = b[ii];
        for (uint32_t k = ii + 1; k < d; k++) v -= L[(size_t)k * d + ii] * b[k];
        b[ii] = v / L[(size_t)ii * d + ii];
    }
}

/* Modified Gram-Schmidt orthonormalization of the k columns of Q (d x k)
 * with reorthogonalization ("twice is enough", Kahan): one float pass leaves
 * cancellation noise ALIGNED with the earlier columns whenever the residual
 * is tiny -- and subspace iteration on a low-rank W drives trailing columns
 * into exactly that regime -- so a second projection pass is mandatory for a
 * usable basis. Columns whose residual collapses below 1e-4 of their
 * pre-projection norm carry no independent direction: re-seed them
 * deterministically (bounded retries; a genuinely exhausted column is left
 * zero, which downstream turns into z=0, a harmless dead direction). */
static void mgs(float *Q, uint32_t d, uint32_t k, uint64_t *seed) {
    for (uint32_t j = 0; j < k; j++) {
        for (int attempt = 0; ; attempt++) {
            double pre = 0.0;
            for (uint32_t i = 0; i < d; i++) {
                double v = Q[(size_t)i * k + j];
                pre += v * v;
            }
            pre = sqrt(pre);
            for (int pass = 0; pass < 2; pass++) {
                for (uint32_t p = 0; p < j; p++) {
                    double dot = 0.0;
                    for (uint32_t i = 0; i < d; i++)
                        dot += (double)Q[(size_t)i * k + p] * Q[(size_t)i * k + j];
                    for (uint32_t i = 0; i < d; i++)
                        Q[(size_t)i * k + j] -= (float)dot * Q[(size_t)i * k + p];
                }
            }
            double nrm = 0.0;
            for (uint32_t i = 0; i < d; i++) {
                double v = Q[(size_t)i * k + j];
                nrm += v * v;
            }
            nrm = sqrt(nrm);
            if (nrm > 1e-4 * pre && nrm > 1e-30) {
                for (uint32_t i = 0; i < d; i++) Q[(size_t)i * k + j] /= (float)nrm;
                break;
            }
            if (attempt >= 8) {                 /* exhausted: dead direction */
                for (uint32_t i = 0; i < d; i++) Q[(size_t)i * k + j] = 0.0f;
                break;
            }
            for (uint32_t i = 0; i < d; i++) Q[(size_t)i * k + j] = lcg_unit(seed);
        }
    }
}

ds4_z *ds4_z_solve(const float *X, const float *R, uint32_t n,
                   uint32_t d_in, uint32_t d_out, uint32_t rank,
                   float lambda) {
    if (!X || !R || n == 0 || d_in == 0 || d_out == 0 || rank == 0) return NULL;
    if (rank > d_in) rank = d_in;
    if (rank > d_out) rank = d_out;

    /* Normal equations in double: A = X^T X + lambda*mean(diag)*I (d_in x
     * d_in), B = X^T R (d_in x d_out). Accumulation order is fixed, so the
     * solve is reproducible. */
    double *A = calloc((size_t)d_in * d_in, sizeof(double));
    double *B = calloc((size_t)d_in * d_out, sizeof(double));
    float  *W = malloc((size_t)d_in * d_out * sizeof(float));
    if (!A || !B || !W) { free(A); free(B); free(W); return NULL; }
    for (uint32_t t = 0; t < n; t++) {
        const float *x = X + (size_t)t * d_in;
        const float *r = R + (size_t)t * d_out;
        for (uint32_t i = 0; i < d_in; i++) {
            const double xi = x[i];
            if (xi == 0.0) continue;
            double *Ai = A + (size_t)i * d_in;
            for (uint32_t j = i; j < d_in; j++) Ai[j] += xi * x[j];
            double *Bi = B + (size_t)i * d_out;
            for (uint32_t j = 0; j < d_out; j++) Bi[j] += xi * r[j];
        }
    }
    /* mirror the upper triangle + dimensionless ridge */
    double tr = 0.0;
    for (uint32_t i = 0; i < d_in; i++) tr += A[(size_t)i * d_in + i];
    const double ridge = (double)lambda * (tr / (double)d_in) + 1e-10;
    for (uint32_t i = 0; i < d_in; i++) {
        A[(size_t)i * d_in + i] += ridge;
        for (uint32_t j = i + 1; j < d_in; j++)
            A[(size_t)j * d_in + i] = A[(size_t)i * d_in + j];
    }
    if (cholesky(A, d_in) != 0) { free(A); free(B); free(W); return NULL; }
    double *col = malloc((size_t)d_in * sizeof(double));
    if (!col) { free(A); free(B); free(W); return NULL; }
    for (uint32_t j = 0; j < d_out; j++) {      /* solve per output column */
        for (uint32_t i = 0; i < d_in; i++) col[i] = B[(size_t)i * d_out + j];
        cholesky_solve(A, d_in, col);
        for (uint32_t i = 0; i < d_in; i++) W[(size_t)i * d_out + j] = (float)col[i];
    }
    free(col);
    free(A);
    free(B);

    /* Rank-k truncation by subspace iteration on W W^T: V converges to the
     * top-k left singular subspace of W, then M = V^T W = diag(z) U^T. */
    ds4_z *zl = calloc(1, sizeof(*zl));
    float *V = malloc((size_t)d_in * rank * sizeof(float));
    float *T = malloc((size_t)d_out * rank * sizeof(float));   /* W^T V */
    float *M = malloc((size_t)rank * d_out * sizeof(float));
    float *U = malloc((size_t)d_out * rank * sizeof(float));
    float *z = malloc((size_t)rank * sizeof(float));
    if (!zl || !V || !T || !M || !U || !z) {
        free(zl); free(V); free(T); free(M); free(U); free(z); free(W);
        return NULL;
    }
    uint64_t seed = 0x5A5A1EEDULL;
    for (size_t i = 0; i < (size_t)d_in * rank; i++) V[i] = lcg_unit(&seed);
    mgs(V, d_in, rank, &seed);
    for (int it = 0; it < 12; it++) {           /* 12 iters: ample for k<<d */
        /* T = W^T V  (d_out x k) */
        memset(T, 0, (size_t)d_out * rank * sizeof(float));
        for (uint32_t i = 0; i < d_in; i++) {
            const float *Wi = W + (size_t)i * d_out;
            const float *Vi = V + (size_t)i * rank;
            for (uint32_t j = 0; j < d_out; j++) {
                const float wij = Wi[j];
                if (wij == 0.0f) continue;
                float *Tj = T + (size_t)j * rank;
                for (uint32_t c = 0; c < rank; c++) Tj[c] += wij * Vi[c];
            }
        }
        /* V = W T  (d_in x k), then re-orthonormalize */
        for (uint32_t i = 0; i < d_in; i++) {
            const float *Wi = W + (size_t)i * d_out;
            float *Vi = V + (size_t)i * rank;
            for (uint32_t c = 0; c < rank; c++) Vi[c] = 0.0f;
            for (uint32_t j = 0; j < d_out; j++) {
                const float wij = Wi[j];
                if (wij == 0.0f) continue;
                const float *Tj = T + (size_t)j * rank;
                for (uint32_t c = 0; c < rank; c++) Vi[c] += wij * Tj[c];
            }
        }
        mgs(V, d_in, rank, &seed);
    }
    /* M = V^T W (k x d_out); z = row norms of M (descending by construction
     * up to iteration accuracy); U rows = normalized M rows. */
    memset(M, 0, (size_t)rank * d_out * sizeof(float));
    for (uint32_t i = 0; i < d_in; i++) {
        const float *Wi = W + (size_t)i * d_out;
        const float *Vi = V + (size_t)i * rank;
        for (uint32_t c = 0; c < rank; c++) {
            const float v = Vi[c];
            if (v == 0.0f) continue;
            float *Mc = M + (size_t)c * d_out;
            for (uint32_t j = 0; j < d_out; j++) Mc[j] += v * Wi[j];
        }
    }
    for (uint32_t c = 0; c < rank; c++) {
        double nrm = 0.0;
        const float *Mc = M + (size_t)c * d_out;
        for (uint32_t j = 0; j < d_out; j++) nrm += (double)Mc[j] * Mc[j];
        nrm = sqrt(nrm);
        z[c] = (float)nrm;
        const float inv = nrm > 1e-20 ? (float)(1.0 / nrm) : 0.0f;
        for (uint32_t j = 0; j < d_out; j++) U[(size_t)j * rank + c] = Mc[j] * inv;
    }
    free(M);
    free(T);
    free(W);

    /* Sort directions by |z| descending so ds4_z_set_rank(k) keeps the top-k
     * (subspace iteration converges the subspace, not the ordering). */
    for (uint32_t a = 0; a < rank; a++) {
        uint32_t best = a;
        for (uint32_t b = a + 1; b < rank; b++) if (z[b] > z[best]) best = b;
        if (best != a) {
            float tz = z[a]; z[a] = z[best]; z[best] = tz;
            for (uint32_t i = 0; i < d_in; i++) {
                float tv = V[(size_t)i * rank + a];
                V[(size_t)i * rank + a] = V[(size_t)i * rank + best];
                V[(size_t)i * rank + best] = tv;
            }
            for (uint32_t j = 0; j < d_out; j++) {
                float tu = U[(size_t)j * rank + a];
                U[(size_t)j * rank + a] = U[(size_t)j * rank + best];
                U[(size_t)j * rank + best] = tu;
            }
        }
    }

    zl->U = U;
    zl->V = V;
    zl->z = z;
    zl->d_in = d_in;
    zl->d_out = d_out;
    zl->rank = rank;
    zl->k = rank;
    return zl;
}

void ds4_z_apply(const ds4_z *zl, const float *x, float *y) {
    if (!zl || !x || !y || zl->k == 0) return;
    const uint32_t k = zl->k <= zl->rank ? zl->k : zl->rank;
    /* t = diag(z) V^T x  (k) */
    float t[512];
    float *tp = k <= 512 ? t : malloc((size_t)k * sizeof(float));
    if (!tp) return;
    for (uint32_t c = 0; c < k; c++) tp[c] = 0.0f;
    for (uint32_t i = 0; i < zl->d_in; i++) {
        const float xi = x[i];
        if (xi == 0.0f) continue;
        const float *Vi = zl->V + (size_t)i * zl->rank;
        for (uint32_t c = 0; c < k; c++) tp[c] += xi * Vi[c];
    }
    for (uint32_t c = 0; c < k; c++) tp[c] *= zl->z[c];
    /* y += U t */
    for (uint32_t j = 0; j < zl->d_out; j++) {
        const float *Uj = zl->U + (size_t)j * zl->rank;
        float acc = 0.0f;
        for (uint32_t c = 0; c < k; c++) acc += Uj[c] * tp[c];
        y[j] += acc;
    }
    if (tp != t) free(tp);
}

void ds4_z_set_rank(ds4_z *zl, uint32_t k) {
    if (!zl) return;
    zl->k = k <= zl->rank ? k : zl->rank;
}

const float *ds4_z_values(const ds4_z *zl, uint32_t *rank_out) {
    if (rank_out) *rank_out = zl ? zl->rank : 0;
    return zl ? zl->z : NULL;
}

/* "DS4Z" v1: header {magic, version, d_in, d_out, rank, k} u32 LE, then
 * U (d_out*rank), V (d_in*rank), z (rank) as f32. */
int ds4_z_save(const ds4_z *zl, const char *path) {
    if (!zl || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    const uint32_t hdr[6] = {0x5A345344u /* "DS4Z" LE */, 1u,
                             zl->d_in, zl->d_out, zl->rank, zl->k};
    int ok = fwrite(hdr, sizeof(hdr), 1, fp) == 1 &&
             fwrite(zl->U, sizeof(float), (size_t)zl->d_out * zl->rank, fp) ==
                 (size_t)zl->d_out * zl->rank &&
             fwrite(zl->V, sizeof(float), (size_t)zl->d_in * zl->rank, fp) ==
                 (size_t)zl->d_in * zl->rank &&
             fwrite(zl->z, sizeof(float), zl->rank, fp) == zl->rank;
    fclose(fp);
    return ok ? 0 : -1;
}

ds4_z *ds4_z_load(const char *path) {
    FILE *fp = path ? fopen(path, "rb") : NULL;
    if (!fp) return NULL;
    uint32_t hdr[6];
    ds4_z *zl = NULL;
    if (fread(hdr, sizeof(hdr), 1, fp) == 1 &&
        hdr[0] == 0x5A345344u && hdr[1] == 1u &&
        hdr[2] && hdr[3] && hdr[4] && hdr[5] <= hdr[4]) {
        zl = calloc(1, sizeof(*zl));
        if (zl) {
            zl->d_in = hdr[2]; zl->d_out = hdr[3];
            zl->rank = hdr[4]; zl->k = hdr[5];
            zl->U = malloc((size_t)zl->d_out * zl->rank * sizeof(float));
            zl->V = malloc((size_t)zl->d_in * zl->rank * sizeof(float));
            zl->z = malloc((size_t)zl->rank * sizeof(float));
            if (!zl->U || !zl->V || !zl->z ||
                fread(zl->U, sizeof(float), (size_t)zl->d_out * zl->rank, fp) !=
                    (size_t)zl->d_out * zl->rank ||
                fread(zl->V, sizeof(float), (size_t)zl->d_in * zl->rank, fp) !=
                    (size_t)zl->d_in * zl->rank ||
                fread(zl->z, sizeof(float), zl->rank, fp) != zl->rank) {
                ds4_z_free(zl);
                zl = NULL;
            }
        }
    }
    fclose(fp);
    return zl;
}

void ds4_z_free(ds4_z *zl) {
    if (!zl) return;
    free(zl->U);
    free(zl->V);
    free(zl->z);
    free(zl);
}
