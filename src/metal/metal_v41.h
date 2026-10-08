/* metal_v41.h — src/metal 里 V4.1 一族 .m 的内部共享头(2026-10-08)。
 * 放这里的是: 暂存槽(grow-only, 带代号)、统一的核发射器、权重范围解析、主机同步。只被 metal_v41_*.m / metal_bwd_*.m / metal_draft_kd.m 包含。 */
#ifndef DS4_METAL_V41_H
#define DS4_METAL_V41_H
#import "metal_internal.h"
#include "metal_v41_args.h"
#include "ds4_gpu_v41.h"
#include "ds4_gpu_bwd.h"
#include "../common/ds4_quantfmt.h"

/* 暂存槽: 只长不缩; 换了缓冲就 +1 代号(ds4_gpu_v41_scratch_generation), 与 CUDA 的 v41_scratch 同语义 */
typedef struct { __strong id<MTLBuffer> buf; uint64_t cap; const char *what; } v41_scratch;
id<MTLBuffer> v41_grow(v41_scratch *s, uint64_t bytes, const char *what);

/* 一次绑定: buf 非空 → setBuffer(off); 否则 bytes/len → setBytes。NULL 张量用 v41_bind_tensor 绑成哑缓冲(Metal 校验不接受空槽)。 */
typedef struct { __strong id<MTLBuffer> buf; NSUInteger off; const void *bytes; NSUInteger len; } v41_bind;
v41_bind v41_bind_tensor(const ds4_gpu_tensor *t);
v41_bind v41_bind_tensor_off(const ds4_gpu_tensor *t, uint64_t byte_off);
v41_bind v41_bind_buf(id<MTLBuffer> b, NSUInteger off);
v41_bind v41_bind_bytes(const void *p, NSUInteger len);
#define V41_T(t) v41_bind_tensor(t)
#define V41_TO(t, off) v41_bind_tensor_off((t), (off))
#define V41_B(b, off) v41_bind_buf((b), (off))
#define V41_A(p) v41_bind_bytes(&(p), sizeof(p))
/* 发射: binds[i] 绑到 buffer(i); grid 按 threadgroup 数给; 返回 1 成功。核没编进库 / 找不到 → 0 并打名字。 */
int v41_launch(const char *kernel, const v41_bind *binds, uint32_t nbind, MTLSize tgs, MTLSize tg);
/* 一维便捷: n 个线程, 每 threadgroup 256 */
int v41_launch_1d(const char *kernel, const v41_bind *binds, uint32_t nbind, uint64_t n);

/* 权重范围 → (视图缓冲, 视图内偏移)。超出映射 / 没有覆盖的视图 → nil(wrap 已打印) */
id<MTLBuffer> v41_model_buf(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t len, uint64_t *inner_off, const char *what);
/* 把当前批里已排的命令提交并等 GPU 跑完, 批保持打开(core 之后还要 end_commands)。主机要读设备结果前调。 */
int v41_host_sync(void);
/* 设备张量清零 / 填 u32 */
int v41_fill_u32(ds4_gpu_tensor *t, uint64_t byte_off, uint64_t n_u32, uint32_t v);
int v41_fill_buf_u32(id<MTLBuffer> b, NSUInteger off, uint64_t n_u32, uint32_t v);

/* 共用的发射片(实现在 metal_v41_dense.m): 任意权重类型的 GEMV(n ≤ 8) 与预填 GEMM; 非分组传 n_groups=1、步长 0 */
int v41_gemv(const void *model_map, uint64_t model_size, uint32_t wtype, uint64_t off, uint64_t in_dim, uint64_t out_dim,
             const ds4_gpu_tensor *x, uint32_t x_stride, ds4_gpu_tensor *out, uint32_t out_stride, uint32_t n_tok,
             uint32_t n_groups, uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out, const ds4_gpu_tensor *skip, const char *what);
/* out[M][N] (+)= x[M][K]·B; wnn=0: B(k,n)=W[n][k](W 是 [N][K]); wnn=1: B(k,n)=W[k][n](W 是 [K][N]) */
int v41_wgemm(const void *model_map, uint64_t model_size, uint32_t wtype, uint64_t off, uint32_t wnn, uint32_t M, uint32_t N, uint32_t K,
              const ds4_gpu_tensor *x, uint32_t lda, ds4_gpu_tensor *out, uint32_t ldc, uint32_t n_groups, uint64_t w_gstride,
              uint32_t x_gstride, uint32_t out_gstride, int beta, int round_out, const char *what);
/* C[M][N] = alpha·op(A)·op(B) + beta·C(f32 行主序; transA: A 存 [K][M]; transB: B 存 [N][K]) */
int v41_sgemm(const ds4_gpu_tensor *A, uint32_t lda, int transA, const ds4_gpu_tensor *B, uint32_t ldb, int transB, ds4_gpu_tensor *C, uint32_t ldc,
              uint32_t M, uint32_t N, uint32_t K, float alpha, float beta, const char *what);
/* f32 GEMV 直发(n ≤ 8, 设备 f32 权重张量 W[out][in]): 草稿 bias 等用 */
int v41_f32_gemv_dev(const ds4_gpu_tensor *W, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, ds4_gpu_tensor *out, uint32_t n_tok, const char *what);

/* 反修增益覆盖表 [layer] → 设备 [n_expert][OUT] f32(实现在 metal_v41_moe.m); 空 = 该层不挂 */
id<MTLBuffer> v41_gr_buf(uint32_t layer);
/* VQ 专家前向的预填骨架(排序 + 分组核), 供反向重算借用: 返回排序后的 ys/inv/meta 等(实现在 metal_v41_moe.m) */
typedef struct { uint32_t nv, nact, npair; id<MTLBuffer> perm, inv, meta; } v41_vq_sort;
int v41_vq_sort_pairs(const ds4_gpu_tensor *selected, uint32_t n_tok, uint32_t K, uint32_t n_total_expert, v41_vq_sort *out);

#endif /* DS4_METAL_V41_H */
