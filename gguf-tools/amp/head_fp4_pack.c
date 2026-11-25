/* head_fp4_pack.c — Build full-rank FP4x32 residual sidecar head_fp4.bin from factory BF16 head.weight minus GGUF q4_K output.weight. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "ds4_st41.h"
#include "ds4_gguf.h"
#include "ds4_quantfmt.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PT124_HEAD_MAGIC 0x34323148u
#define PT124_VER        1u

typedef struct {
    uint32_t magic, version, rows, cols;
    float alpha;
    uint32_t pad[3];
    uint64_t total_bytes;
    uint64_t fnv64;
    uint64_t reserved[2];
} head_hdr;

typedef char head_hdr_sz_check[sizeof(head_hdr) == 64 ? 1 : -1];

static inline float bf16_to_f32(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float f;
    memcpy(&f, &w, 4);
    return f;
}

static void drop_pages(const void *ptr, uint64_t nbytes) {
    const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    uintptr_t a = ((uintptr_t)ptr + pg - 1u) & ~(pg - 1u);
    uintptr_t b = ((uintptr_t)ptr + nbytes) & ~(pg - 1u);
    if (b > a) (void)madvise((void *)a, (size_t)(b - a), MADV_DONTNEED);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <hf_dir> <gguf_path> <out_bin>\n", argv[0]);
        return 2;
    }
    const uint64_t ROWS = 129280ull, COLS = 5120ull;
    const uint64_t ROW_Q4K = (COLS / 256ull) * 144ull;
    const uint64_t ROW_FP4 = (COLS / 32ull) * 17ull;
    const uint64_t PAYLOAD = ROWS * ROW_FP4;

    v41_st S;
    if (v41_st_open(&S, argv[1]) != 0) return 1;
    ds4_gguf G;
    char err[256] = {0};
    if (ds4_gguf_open(&G, argv[2], err, sizeof err) != 0) {
        fprintf(stderr, "gguf open failed: %s\n", err);
        return 1;
    }
    const v41_st_ent *ew = v41_st_find(&S, "head.weight");
    const ds4_gguf_tensor *gt = ds4_gguf_find(&G, "output.weight");
    if (!ew || !gt) { fprintf(stderr, "missing head.weight or output.weight\n"); return 1; }
    if ((uint64_t)v41_st_numel(ew) != ROWS * COLS || ew->nd != 2 || strcmp(ew->dtype, "BF16") != 0 ||
        (uint64_t)ew->shape[0] != ROWS || (uint64_t)ew->shape[1] != COLS) {
        fprintf(stderr, "bad HF head.weight shape/dtype\n");
        return 1;
    }
    if (gt->type != DS4_GGT_Q4_K || gt->ne[0] != COLS || gt->ne[1] != ROWS) {
        fprintf(stderr, "bad GGUF output.weight type=%u\n", gt->type);
        return 1;
    }
    uint64_t gbytes = 0;
    const uint16_t *pw = (const uint16_t *)v41_st_data(&S, ew);
    const uint8_t *pg = ds4_gguf_tensor_data(&G, gt, &gbytes);
    if (!pw || !pg) return 1;

    FILE *fo = fopen(argv[3], "wb");
    if (!fo) { perror("fopen out"); return 1; }
    head_hdr H;
    memset(&H, 0, sizeof H);
    H.magic = PT124_HEAD_MAGIC; H.version = PT124_VER;
    H.rows = (uint32_t)ROWS; H.cols = (uint32_t)COLS;
    H.alpha = 1.0f; H.total_bytes = 64ull + PAYLOAD;
    if (fwrite(&H, 1, sizeof H, fo) != sizeof H) return 1;

    const uint64_t CHUNK_R = 4096ull;
    uint8_t *pk = (uint8_t *)malloc((size_t)(CHUNK_R * ROW_FP4));
    if (!pk) return 1;

    double sse0 = 0.0, sse1 = 0.0, en = 0.0;
    int bad_nan = 0;
    uint64_t fnv = 14695981039346656037ull;

    for (uint64_t r0 = 0; r0 < ROWS; r0 += CHUNK_R) {
        const uint64_t nr = (ROWS - r0 < CHUNK_R) ? (ROWS - r0) : CHUNK_R;
        double c_sq0 = 0.0, c_sq1 = 0.0, c_se = 0.0;
        int c_nan = 0;

#ifdef _OPENMP
        #pragma omp parallel for reduction(+:c_sq0,c_sq1,c_se,c_nan) schedule(static)
#endif
        for (uint64_t dr = 0; dr < nr; dr++) {
            const uint64_t r = r0 + dr;
            const uint16_t *bf_row = pw + r * COLS;
            const uint8_t *q4_row = pg + r * ROW_Q4K;
            uint8_t *pk_row = pk + dr * ROW_FP4;
            for (uint64_t b = 0; b < COLS / 256ull; b++) {
                float q4[256], res[256], rec[256];
                ds4_deq_q4_K(q4_row + b * 144ull, 1u, q4);
                for (uint32_t j = 0; j < 256u; j++) {
                    float w_fp = bf16_to_f32(bf_row[b * 256ull + j]);
                    if (!isfinite(w_fp)) { c_nan++; w_fp = 0.0f; }
                    const float d0 = w_fp - q4[j];
                    res[j] = d0;
                    c_sq0 += (double)d0 * (double)d0;
                    c_se  += (double)w_fp * (double)w_fp;
                }
                ds4_quant_fp4x32(res, 8u, pk_row + b * 8ull * 17ull);
                ds4_deq_fp4x32(pk_row + b * 8ull * 17ull, 8u, rec);
                for (uint32_t j = 0; j < 256u; j++) {
                    const float w_fp = bf16_to_f32(bf_row[b * 256ull + j]);
                    const double d1 = (double)(q4[j] + rec[j]) - (double)w_fp;
                    c_sq1 += d1 * d1;
                }
            }
        }
        sse0 += c_sq0; sse1 += c_sq1; en += c_se; bad_nan += c_nan;
        const size_t nb = (size_t)(nr * ROW_FP4);
        for (size_t i = 0; i < nb; i += 64) { fnv ^= pk[i]; fnv *= 1099511628211ull; }
        if (fwrite(pk, 1, nb, fo) != nb) { perror("fwrite"); return 1; }
        drop_pages(pw + r0 * COLS, nr * COLS * 2ull);
        drop_pages(pg + r0 * ROW_Q4K, nr * ROW_Q4K);
    }
    free(pk);
    if (bad_nan > 0) { fprintf(stderr, "NaN in head.weight: %d\n", bad_nan); fclose(fo); return 1; }
    H.fnv64 = fnv;
    if (fseek(fo, 0, SEEK_SET) != 0 || fwrite(&H, 1, sizeof H, fo) != sizeof H || fclose(fo) != 0) return 1;
    v41_st_release_idle(&S);
    v41_st_close(&S);
    ds4_gguf_close(&G);
    fprintf(stderr, "[head_fp4_pack] rows=%llu cols=%llu bytes=%llu rel_rmse: %.5f -> %.5f (var_rec=%.2f%%) fnv=%016llx\n",
            (unsigned long long)ROWS, (unsigned long long)COLS, (unsigned long long)H.total_bytes,
            sqrt(sse0 / en), sqrt(sse1 / en), (1.0 - sse1 / sse0) * 100.0, (unsigned long long)fnv);
    return 0;
}
