/* metal_v41_args.h — V4.1 Metal 核的参数块(主机侧镜像; 与 metal/v41_*.metal 里的同名 struct 逐字段同序同型)。
 * shader 源是纯文本拼接、不能 #include, 所以这里是手抄的第二份 —— 改一边必须改另一边, 漂开的症状是核读到错位的参数、
 * 输出悄悄变垃圾而不报错(与 metal_args.h 对老核的约定相同)。字段只用 uint32/float/uint64, uint64 全放在最前面保证 8 对齐。 */
#ifndef DS4_METAL_V41_ARGS_H
#define DS4_METAL_V41_ARGS_H
#include <stdint.h>

enum { V41_WT_FP4X32 = 0, V41_WT_Q4K = 1, V41_WT_BF16 = 2, V41_WT_F32 = 3, V41_WT_FP8BLK = 4 };

typedef struct { uint32_t n0, n1, n2, n3; float f0, f1, f2, f3; } v41_n_args;
typedef struct {
    uint64_t w_off, w_gstride, sc_off, sc_gstride;
    uint32_t in_dim, out_dim, x_stride, out_stride, ksplit, x_gstride, out_gstride, n_tok, round_out, wtype, sbc, has_skip;
} v41_gemv_args;
typedef struct {
    uint64_t w_off, w_gstride, sc_off, sc_gstride;
    uint32_t M, N, K, lda, ldc, x_gstride, out_gstride, wtype, wnn, wcols, sbc, beta, round_out, pad0;
} v41_wgemm_args;
typedef struct { uint32_t M, N, K, lda, ldb, ldc, transA, transB; float alpha, beta; uint32_t a_gstride, b_gstride, c_gstride, pad1; } v41_sgemm_args;
typedef struct { uint32_t E, n_hc, n_rows, src_row0, slot, n_slot, dst_pos0, cap, has_posd, pad; } v41_hcmean_args;
typedef struct { uint32_t n_tok, n_expert, topk, pad; float route_scale; float pad1, pad2, pad3; } v41_router_args;
typedef struct { uint32_t ratio, dim, n, has_snap; } v41_cstep_args;
typedef struct { uint32_t n_head, head_dim, n_rot, osl, inverse, do_round; float theta, factor, beta_fast, beta_slow; } v41_rope_args;
typedef struct { uint32_t g0, dim, blk, row_bytes, nib_bytes, nb_row, mode, has_posd, ratio, g_trash, nbatch, pad; } v41_kvpack_args;
typedef struct { uint32_t pos0, i0, window, hd, back, has_posd, pad0, pad1; } v41_ring_args;
typedef struct { uint32_t pos0, window, ng, topk, n_head, hd, full_block, ring, win_lo, has_posd, has_comp, pad; float scale; float pad1, pad2, pad3; } v41_attn_args;
typedef struct { uint32_t pos0, ng, n_head, dk, ratio, has_posd, has_cand, cand_bs, cand_cap, n_rows, topk, pad; } v41_idx_args;
typedef struct { uint32_t pos0, ng, ratio, topk_blocks, bs, has_posd, n_rows, nb_cap; } v41_cand_args;
typedef struct { uint64_t seed; uint32_t V, row0, n_rows, top_k, stream, has_q; float inv_T, min_p, top_p; uint32_t pad0, pad1, pad2; } v41_sample_args;
typedef struct { uint32_t IN, MID, OUT, K, v3, has_gov, n_tok, n_act; float clamp_v; uint32_t pad0, pad1, pad2; } v41_vq_args;
typedef struct { uint32_t IN, MID, OUT, K, n_expert, cur_view, pad0, pad1; float clamp_v; float pad2, pad3, pad4; } v41_mtp_args;
typedef struct { uint32_t row0, V, K, has_w; float scale; float pad0, pad1, pad2; } v41_kl_args;
typedef struct { uint64_t n; float lr, b1, b2, eps, gs, c1, c2, pad; } v41_adam_args;
typedef struct { uint32_t window, ng, topk, n_head, hd, has_comp, has_gcomp, pad; float scale; float pad1, pad2, pad3; } v41_battn_args;
typedef struct { uint32_t n_tok, hc, iters, mh, dim, has_gpre, has_gpost, has_gcomb; float eps, norm_eps, pad0, pad1; } v41_bhc_args;
typedef struct { uint32_t NE, K, pad0, pad1; float rs; float pad2, pad3, pad4; } v41_brt_args;
typedef struct { uint32_t which, R, C, n_act, v3, has_gov, OUTd, accumulate; uint32_t do_round, pad0, pad1, pad2; } v41_bvq_args;
typedef struct { uint32_t hbase, B, window, n_head, hd, nb, pad0, pad1; float scale; float pad2, pad3, pad4; } v41_dk_args;
typedef struct { uint32_t V, K, pad0, pad1; float T, scale, pad2, pad3; } v41_tv_args;

#endif /* DS4_METAL_V41_ARGS_H */
