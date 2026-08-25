/* hf_read.h — minimal, self-contained safetensors reader for the original
 * DeepSeek-V4-Flash HF checkpoint (sharded, F16/BF16/F32/F8_E4M3 + block scale).
 *
 * RAM policy: a tensor's [begin,end) byte slice is fseek+fread on demand from
 * its shard; a whole shard is NEVER loaded.  Only the (small) JSON header of a
 * shard is read into RAM, lazily, the first time that shard is touched.
 *
 * This reader does not link against deepseek4-quantize.c; the dtype-conversion
 * and FP8 block-dequant math is replicated here. */
#ifndef GO_ONEBIT_HF_READ_H
#define GO_ONEBIT_HF_READ_H

#include <stdint.h>

typedef struct hf_db hf_db;

/* Parse <hf_dir>/model.safetensors.index.json (or, as a fallback, a single
 * <hf_dir>/model.safetensors).  Shard headers are read lazily.  NULL on error. */
hf_db *hf_open(const char *hf_dir);

/* 1 if tensor exists in the weight map, 0 otherwise.  No shard/header read. */
int hf_has(hf_db *db, const char *name);

/* Fill shape/ndim/dtype WITHOUT reading any tensor data (reads only the owning
 * shard's JSON header, lazily).  dtype_out holds e.g. "F16"/"BF16"/"F32"/
 * "F8_E4M3".  Returns 0 on success, -1 on error. */
int hf_meta(hf_db *db, const char *name, int64_t shape_out[8], int *ndim_out,
            char dtype_out[16]);

/* Read exactly this tensor's byte slice from its shard and convert to f32.
 * F16/BF16/F32 convert element-wise.  F8_E4M3 additionally reads the companion
 * block-scale tensor (<name with .weight -> .scale>) and applies the 128x128
 * block dequant.  Returns a malloc'd buffer the caller frees; *n_out = element
 * count.  NULL on error. */
float *hf_read_f32(hf_db *db, const char *name, int64_t *n_out);

void hf_close(hf_db *db);

#endif /* GO_ONEBIT_HF_READ_H */
