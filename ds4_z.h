/* =========================================================================
 * ds4_z -- z 隐变量: closed-form low-rank latent correction, as a
 * self-contained module (no engine/graph/tensor coupling).
 * =========================================================================
 *
 * The latent-variable product of the go-onebit scheme (ALGORITHM.md §6.2,
 * "三段式产物" ③): a per-layer low-rank map that predicts the residual the
 * quantized base gets wrong, solved CLOSED-FORM from calibration activations
 * (zero training), with the rank k_L adjustable per layer AFTER solving --
 * the singular directions are ordered, so truncating z is the quality/size
 * dial.
 *
 *   y_corrected = y_base + U * diag(z[0..rank)) * V^T * x
 *
 * Solve pipeline (ds4_z_solve): ridge normal equations
 * (X^T X + lambda I) W = X^T R  ->  Cholesky  ->  rank-k truncation of W by
 * subspace (orthogonal) iteration. Deterministic: fixed-seed LCG init, no
 * global RNG state, so a re-solve is byte-reproducible. */
#ifndef DS4_Z_H
#define DS4_Z_H

#include <stdint.h>

/* One solved latent: W (d_in x d_out) factored as V(d_in x rank) *
 * diag(z) * U^T(rank x d_out). Row-major throughout. */
typedef struct {
    float   *U;      /* d_out x rank (column j = output direction j) */
    float   *V;      /* d_in  x rank */
    float   *z;      /* rank singular values -- THE adjustable latent */
    uint32_t d_in;
    uint32_t d_out;
    uint32_t rank;   /* allocated rank */
    uint32_t k;      /* active rank <= rank (adjustable per layer) */
} ds4_z;

/* Closed-form solve from calibration pairs: X (n x d_in) inputs, R
 * (n x d_out) residual targets (y_ref - y_base). lambda is the ridge
 * regularizer (>= 0; scaled by the mean diagonal of X^T X so it is
 * dimensionless). Returns a solved latent with k == rank, or NULL on
 * OOM/degenerate input. O(n*d^2 + d^3): an OFFLINE post-train step. */
ds4_z *ds4_z_solve(const float *X, const float *R, uint32_t n,
                   uint32_t d_in, uint32_t d_out, uint32_t rank,
                   float lambda);

/* y += U diag(z[0..k)) V^T x -- the runtime-side apply. O(k*(d_in+d_out)). */
void ds4_z_apply(const ds4_z *zl, const float *x, float *y);

/* The per-layer dial: use only the top-k directions. k > rank clamps. */
void ds4_z_set_rank(ds4_z *zl, uint32_t k);

/* Residual energy captured per direction (z[i]^2, descending): lets a
 * caller pick k_L per layer against a quality budget. */
const float *ds4_z_values(const ds4_z *zl, uint32_t *rank_out);

/* Flat binary serialization (magic "DS4Z", version 1). */
int ds4_z_save(const ds4_z *zl, const char *path);
ds4_z *ds4_z_load(const char *path);

void ds4_z_free(ds4_z *zl);

#endif /* DS4_Z_H */
