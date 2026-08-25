/* npy.h — minimal NumPy .npy reader for Go-domain calibration activations.
 *
 * Reads C-order (row-major) little-endian .npy files (v1.0 and v2.0 headers).
 * Supported element dtypes for npy_read_f32: float16 (<f2), float32 (<f4),
 * float64 (<f8), signed ints (i1/i2/i4/i8), unsigned ints (u1/u2/u4/u8),
 * bool (b1). Big-endian ('>') and fortran_order True are rejected.
 *
 * Pure C99. No external deps.
 */
#ifndef GO_ONEBIT_NPY_H
#define GO_ONEBIT_NPY_H

#include <stdint.h>

typedef struct {
    int     ndim;        /* number of dimensions (0..8) */
    int64_t shape[8];    /* extent of each dimension */
    char    kind;        /* numpy kind char: 'f' float, 'i' int, 'u' uint, 'b' bool */
    int     itemsize;    /* bytes per element */
    int     fortran;     /* 1 if fortran_order (column-major) — rejected by reader */
    int64_t count;       /* total element count = product of shape (1 for 0-d) */
} npy_meta;

/* Parse header only; no array data is read.
 * Returns 0 on success, -1 on error. *out is filled on success. */
int npy_stat(const char *path, npy_meta *out);

/* Read the whole array, converting every element to float32:
 *   f2 -> exact IEEE half->float, f4 -> copy, f8 -> narrow, ints -> cast.
 * Returns a malloc'd float[count] the caller must free(), or NULL on error.
 * If out != NULL it is filled with the array metadata. */
float *npy_read_f32(const char *path, npy_meta *out);

#endif /* GO_ONEBIT_NPY_H */
