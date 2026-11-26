/* core_v41_exit.h — Stage B 出口残差(head_fp4.bin)与双项叠加平滑出口补偿(exit_ctrl.bin)接口(2026-11-26, Round 126/144)。
 *
 * pt124_exit_hdr.flags 位定义:
 *   1u  : 含 f32 对角缩放向量 dg[E]
 *   4u  : A[R,E], B[R,E] 以 IEEE f16 存储(未置位则为 f32)
 *   16u : 启用主子空间解耦项(R2 = pad[0], pad[1] 存 f32 alpha_perp, 尾部附 f16 B2[R2,E])
 *   32u : (配合 16u) B2 后附 f32 逐模态谱权重 w2[R2], A2[r,:] = w2[r]·B2[r,:]
 *   64u : (配合 16u) B2 前显式附 f16 A2[R2,E] */
#ifndef DS4_CORE_V41_EXIT_H
#define DS4_CORE_V41_EXIT_H

#include "core_internal.h"

#define PT124_HEAD_MAGIC 0x34323148u   /* 'H124' */
#define PT124_EXIT_MAGIC 0x34323145u   /* 'E124' */
#define PT124_VER        1u
#define PT124_MAX_RANK   1024u

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
