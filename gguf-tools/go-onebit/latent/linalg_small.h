/* linalg_small.h — tiny double-precision dense linear algebra for the
 * Go-domain 1-bit quantizer's four-loss closed-form solve (SPEC.md §7).
 *
 * Two public routines, both written from scratch (this repo links no
 * LAPACK/Eigen/GGML and has no other SVD/Cholesky/eigen code):
 *
 *   - chol_solve_spd : symmetric-positive-definite linear solve (normal
 *     equations) via Cholesky factorization. Used for the routing /
 *     smoothness / alignment least-squares terms.
 *   - sym_eig_topk   : deterministic top-k symmetric eigenpairs, used to
 *     truncate the per-layer correction map to rank d_ℓ.
 *
 * Pure C99. All matrices are row-major. No global state, no hidden RNG:
 * sym_eig_topk's randomness is fully determined by its `seed` argument.
 */
#ifndef LINALG_SMALL_H
#define LINALG_SMALL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Cholesky-factorize the symmetric positive-definite matrix A (n×n,
 * row-major) in place — on success the lower triangle of A is overwritten
 * with the Cholesky factor L (A = L·Lᵀ) — then solve A·X = B for X, where B
 * is n×nrhs row-major; the solution overwrites B.
 *
 * Only the lower triangle (incl. diagonal) of A is read, so a symmetric
 * matrix supplied in full works as-is.
 *
 * Returns 0 on success, nonzero if A is not positive definite within
 * working precision (a nonpositive pivot is encountered). On failure B is
 * left partially modified and must be considered invalid.
 *
 * NOTE: the four-loss "fixed/noise" term is a Tikhonov ridge — callers add
 * λI to A (the normal-equations Gram matrix) BEFORE calling this, both to
 * pin the otherwise underdetermined solution and to guarantee positive
 * definiteness. If this still returns nonzero, increase λ and retry. */
int chol_solve_spd(double *A, int n, double *B, int nrhs);

/* Compute the top-k eigenpairs of the symmetric n×n matrix M (row-major)
 * by power iteration with Gram-Schmidt (Hotelling) deflation.
 *
 * Outputs (caller-allocated):
 *   evecs : k×n row-major; row i is a unit eigenvector.
 *   evals : k eigenvalues, sorted descending; evals[i] pairs with row i.
 *
 * `iters` bounds the power-iteration steps per eigenvector (<=0 picks a
 * sensible default). Convergence is geometric in the eigenvalue gap ratio.
 *
 * DETERMINISTIC: the start vectors are drawn from a splitmix64 PRNG seeded
 * solely by `seed`; for identical (M,n,k,iters,seed) the result is
 * bit-identical. No rand()/random()/time() is used.
 *
 * The k pairs of largest magnitude are returned (these are the most
 * significant components for rank-d truncation of the correction map; for
 * the PSD operators in this pipeline they coincide with the algebraically
 * largest). k is clamped to n if larger. */
void sym_eig_topk(const double *M, int n, int k,
                  double *evecs, double *evals, int iters, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* LINALG_SMALL_H */
