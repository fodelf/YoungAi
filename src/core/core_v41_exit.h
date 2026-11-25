/* core_v41_exit.h — Stage B 出口残差(head_fp4.bin)与平滑出口补偿(exit_ctrl.bin)接口(2026-11-25, Round 126)。 */
#ifndef DS4_CORE_V41_EXIT_H
#define DS4_CORE_V41_EXIT_H

#include "core_internal.h"

#define PT124_HEAD_MAGIC 0x34323148u   /* 'H124' */
#define PT124_EXIT_MAGIC 0x34323145u   /* 'E124' */
#define PT124_VER        1u
#define PT124_MAX_RANK   512u

typedef struct {
    uint32_t magic, version, rows, cols;
    float alpha;
    uint32_t pad[3];
    uint64_t total_bytes;
    uint64_t fnv64;
    uint64_t reserved[2];
} pt124_head_hdr;

typedef struct {
    uint32_t magic, version, rank, flags;
    float alpha_head;
    uint32_t pad[3];
    uint64_t total_bytes;
    uint64_t reserved[3];
} pt124_exit_hdr;

typedef char pt124_head_hdr_chk[sizeof(pt124_head_hdr) == 64 ? 1 : -1];
typedef char pt124_exit_hdr_chk[sizeof(pt124_exit_hdr) == 64 ? 1 : -1];

#ifndef DS4_NO_GPU
int  v41_exit_load(const char *dir, double *mb_io);
int  v41_exit_save(const char *dir);
void v41_exit_clear(void);
bool v41_exit_active(void);
bool v41_exit_apply(const ds4_model *mdl, const ds4_tensor *w_out,
                    ds4_gpu_tensor *logits, const ds4_gpu_tensor *xn, uint32_t n);
bool v41_exit_bwd_proj(const ds4_model *mdl, const ds4_tensor *w_out,
                       ds4_gpu_tensor *g_xn, const ds4_gpu_tensor *glog, uint32_t m);
#endif

#endif
