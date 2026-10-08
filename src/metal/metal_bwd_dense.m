/* metal_bwd_dense.m — 后训练反传稠密原语的 Metal 发射(2026-10-08): 转置乘 / 教师 top-K / KL 梯度 / RMSNorm·hc_pre·hc_post 反向 /
 * 低秩放大器反向 / Σg² / Adam / bf16 存档。契约 ds4_gpu_bwd.h; 核在 metal/v41_bwd.metal + v41_dense.metal(wgemm/sgemm)。
 * 转置乘走 wgemm(wnn=1), 权重边解边乘, 值与前向同一张表; 层内 bf16 权重缓存(wcache)与专家截留(moe_capture)在 Metal 上是空操作
 * (没有"解成 bf16 落暂存"这一步, 也就没有可缓存的东西; 反传照常重算)。 */
#import "metal_v41.h"

static uint32_t v41_wt_from_ggt(uint32_t wtype) {
    switch (wtype) {
        case DS4_GGT_F32: return V41_WT_F32;
        case DS4_GGT_Q4_K: return V41_WT_Q4K;
        case DS4_GGT_BF16: return V41_WT_BF16;
        case DS4_GGT_FP4X32: return V41_WT_FP4X32;
        case DS4_GGT_FP8_32X32: return V41_WT_FP8BLK;
        default: fprintf(stderr, "ds4: [bwd-metal] 转置乘不认权重类型 %u\n", wtype); return 0xFFFFFFFFu;
    }
}
static uint64_t v41_wbytes_rc(uint32_t wt, uint64_t rows, uint64_t cols) {
    switch (wt) {
        case V41_WT_FP4X32: return rows * (cols / 32u) * 17u;
        case V41_WT_Q4K: return rows * (cols / 256u) * 144u;
        case V41_WT_BF16: return rows * cols * 2u;
        case V41_WT_F32: return rows * cols * 4u;
        default: return rows * cols;
    }
}
int ds4_gpu_bwd_matmul_t_tensor(ds4_gpu_tensor *gx, const void *model_map, uint64_t model_size, uint32_t wtype, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                const ds4_gpu_tensor *gy, uint32_t n_tok, int accumulate) {
    if (!gx || !gy || n_tok == 0) return 0;
    if (ds4_gpu_tensor_bytes(gx) < (uint64_t)n_tok * in_dim * 4 || ds4_gpu_tensor_bytes(gy) < (uint64_t)n_tok * out_dim * 4) return 0;
    const uint32_t wt = v41_wt_from_ggt(wtype);
    if (wt == 0xFFFFFFFFu) return 0;
    return v41_wgemm(model_map, model_size, wt, weight_offset, 1u, n_tok, (uint32_t)in_dim, (uint32_t)out_dim, gy, (uint32_t)out_dim, gx, (uint32_t)in_dim, 1u, 0, 0, 0,
                     accumulate, 0, "bwd matmul_t");
}
int ds4_gpu_bwd_grouped_matmul_t_tensor(ds4_gpu_tensor *gheads, const void *model_map, uint64_t model_size, uint32_t wtype, uint64_t weight_offset, uint32_t n_groups,
                                        uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *glow, uint32_t n_tok, int accumulate) {
    if (!gheads || !glow || n_tok == 0 || wtype == DS4_GGT_F32 || n_groups == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (ds4_gpu_tensor_bytes(gheads) < (uint64_t)n_tok * in_all * 4 || ds4_gpu_tensor_bytes(glow) < (uint64_t)n_tok * out_all * 4) return 0;
    const uint32_t wt = v41_wt_from_ggt(wtype);
    if (wt == 0xFFFFFFFFu) return 0;
    if (wt == V41_WT_FP8BLK && (rank % 32u)) return 0;   /* 缩放按 32 行一格, 组起点必须整格 */
    return v41_wgemm(model_map, model_size, wt, weight_offset, 1u, n_tok, (uint32_t)group_dim, (uint32_t)rank, glow, (uint32_t)out_all, gheads, (uint32_t)in_all,
                     n_groups, v41_wbytes_rc(wt, rank, group_dim), (uint32_t)rank, (uint32_t)group_dim, accumulate, 0, "bwd wo_a matmul_t");
}
int ds4_gpu_bwd_kl_topk_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t row0, uint32_t m, uint32_t n_vocab,
                               const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp, const ds4_gpu_tensor *trest, const ds4_gpu_tensor *w, uint32_t k, float scale) {
    if (!glogits || !loss || !logits || !tid || !tp || !trest || m == 0 || k == 0 || k > 128u || n_vocab > 4096u * 32u) return 0;
    if (ds4_gpu_tensor_bytes(logits) < (uint64_t)(row0 + m) * n_vocab * 4 || ds4_gpu_tensor_bytes(glogits) < (uint64_t)m * n_vocab * 4) return 0;
    if (w && ds4_gpu_tensor_bytes(w) < (uint64_t)m * 4) return 0;
    v41_kl_args a = { row0, n_vocab, k, w ? 1u : 0u, scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(glogits), V41_T(loss), V41_T(logits), V41_T(tid), V41_T(tp), V41_T(trest), V41_T(w) };
    return v41_launch("kernel_v41_bwd_kl_topk", b, 8, MTLSizeMake(m, 1, 1), MTLSizeMake(1024, 1, 1));
}
int ds4_gpu_bwd_topk_tensor(ds4_gpu_tensor *tid, ds4_gpu_tensor *tp, ds4_gpu_tensor *trest, ds4_gpu_tensor *logits, uint32_t row0, uint32_t m, uint32_t n_vocab, uint32_t k) {
    if (!tid || !tp || !trest || !logits || m == 0 || k == 0) return 0;
    if (ds4_gpu_tensor_bytes(logits) < (uint64_t)(row0 + m) * n_vocab * 4 || ds4_gpu_tensor_bytes(tid) < (uint64_t)m * k * 4 || ds4_gpu_tensor_bytes(tp) < (uint64_t)m * k * 4) return 0;
    v41_kl_args a = { row0, n_vocab, k, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(tid), V41_T(tp), V41_T(trest), V41_T(logits) };
    return v41_launch("kernel_v41_bwd_topk", b, 5, MTLSizeMake(m, 1, 1), MTLSizeMake(1024, 1, 1));
}
int ds4_gpu_bwd_rms_norm_tensor(ds4_gpu_tensor *gx, const ds4_gpu_tensor *gxn, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                uint32_t dim, uint32_t n_tok, float eps, int accumulate) {
    if (!gx || !gxn || !x || n_tok == 0) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> wb = v41_model_buf(model_map, model_size, weight_offset, (uint64_t)dim * 4, &inner, "bwd norm w");
    if (!wb) return 0;
    v41_n_args a = { dim, accumulate ? 1u : 0u, 0, 0, eps, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(gx), V41_T(gxn), V41_T(x), V41_B(wb, 0), V41_A(inner) };
    return v41_launch("kernel_v41_bwd_rms_norm", b, 6, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(512, 1, 1));
}
int ds4_gpu_bwd_hc_pre_tensor(ds4_gpu_tensor *ghc, ds4_gpu_tensor *gpre, const ds4_gpu_tensor *gx, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre,
                              uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!ghc || !gpre || !gx || !hc || !pre || n_hc > 4u || n_tok == 0) return 0;
    v41_n_args a = { n_embd, n_hc, 0, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(ghc), V41_T(gpre), V41_T(gx), V41_T(hc), V41_T(pre) };
    return v41_launch("kernel_v41_bwd_hc_pre", b, 6, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(512, 1, 1));
}
int ds4_gpu_bwd_hc_post_tensor(ds4_gpu_tensor *gy, ds4_gpu_tensor *gres, ds4_gpu_tensor *gpost, ds4_gpu_tensor *gcomb, const ds4_gpu_tensor *gout, const ds4_gpu_tensor *y,
                               const ds4_gpu_tensor *res, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!gy || !gout || !y || !res || !post || !comb || n_hc > 4u || n_tok == 0) return 0;
    v41_n_args a = { n_embd, n_hc, (gres ? 1u : 0u) | (gpost ? 2u : 0u) | (gcomb ? 4u : 0u), 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(gy), V41_T(gres), V41_T(gpost), V41_T(gcomb), V41_T(gout), V41_T(y), V41_T(res), V41_T(post), V41_T(comb) };
    return v41_launch("kernel_v41_bwd_hc_post", b, 10, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(512, 1, 1));
}
/* 低秩放大器反向: T = x·Bᵀ, gT = gy·Aᵀ; gA += Tᵀ·gy, gB += gTᵀ·x; gx += gT·B(A/B 存 [K][D]) */
int ds4_gpu_bwd_amp_tensor(ds4_gpu_tensor *gA, ds4_gpu_tensor *gB, ds4_gpu_tensor *gx, const ds4_gpu_tensor *gy, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A,
                           const ds4_gpu_tensor *B, ds4_gpu_tensor *T, ds4_gpu_tensor *gT, uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!gA || !gB || !gy || !x || !A || !B || !T || !gT || n_tok == 0) return 0;
    if (!v41_sgemm(x, D, 0, B, D, 1, T, K, n_tok, K, D, 1.0f, 0.0f, "bwd amp T")) return 0;
    if (!v41_sgemm(gy, D, 0, A, D, 1, gT, K, n_tok, K, D, 1.0f, 0.0f, "bwd amp gT")) return 0;
    if (!v41_sgemm(T, K, 1, gy, D, 0, gA, D, K, D, n_tok, 1.0f, 1.0f, "bwd amp gA")) return 0;
    if (!v41_sgemm(gT, K, 1, x, D, 0, gB, D, K, D, n_tok, 1.0f, 1.0f, "bwd amp gB")) return 0;
    if (gx && !v41_sgemm(gT, K, 0, B, D, 0, gx, D, n_tok, D, K, 1.0f, 1.0f, "bwd amp gx")) return 0;
    return 1;
}
static v41_scratch g_bwd_part;
int ds4_gpu_bwd_sumsq_tensor(const ds4_gpu_tensor *g, uint64_t n, double *out) {
    if (!g || !out) return 0;
    uint64_t ntg = (n + 511u) / 512u;
    if (ntg > 256u) ntg = 256u;
    if (ntg == 0) { *out = 0.0; return 1; }
    id<MTLBuffer> part = v41_grow(&g_bwd_part, ntg * 4, "bwd sumsq");
    if (!part) return 0;
    v41_bind b[] = { V41_B(part, 0), V41_T(g), V41_A(n) };
    if (!v41_launch("kernel_v41_bwd_sumsq", b, 3, MTLSizeMake((NSUInteger)ntg, 1, 1), MTLSizeMake(512, 1, 1))) return 0;
    if (!v41_host_sync()) return 0;
    const float *p = [part contents];
    double s = 0.0;
    for (uint64_t i = 0; i < ntg; i++) s += (double)p[i];
    *out = s;
    return 1;
}
int ds4_gpu_bwd_adam_tensor(ds4_gpu_tensor *p, ds4_gpu_tensor *g, ds4_gpu_tensor *m, ds4_gpu_tensor *v, uint64_t n, float lr, float beta1, float beta2, float eps, float gs, uint32_t step) {
    if (!p || !g || !m || !v || step == 0) return 0;
    v41_adam_args a = { n, lr, beta1, beta2, eps, gs, 1.0f - powf(beta1, (float)step), 1.0f - powf(beta2, (float)step), 0 };
    v41_bind b[] = { V41_A(a), V41_T(p), V41_T(g), V41_T(m), V41_T(v) };
    return v41_launch_1d("kernel_v41_bwd_adam", b, 5, n);
}
int ds4_gpu_bwd_axpy_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, float a, uint64_t n) {
    if (!y || !x || n == 0) return 0;
    v41_bind b[] = { V41_T(y), V41_T(x), V41_A(n), V41_A(a) };
    return v41_launch_1d("kernel_v41_axpy", b, 4, n);
}
int ds4_gpu_bwd_pack_bf16_tensor(ds4_gpu_tensor *dst16, const ds4_gpu_tensor *src, uint64_t n) {
    if (!dst16 || !src || ds4_gpu_tensor_bytes(dst16) < n * 2u || ds4_gpu_tensor_bytes(src) < n * 4u) return 0;
    v41_bind b[] = { V41_T(dst16), V41_T(src), V41_A(n) };
    return v41_launch_1d("kernel_v41_pack_bf16", b, 3, n);
}
int ds4_gpu_bwd_unpack_bf16_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src16, uint64_t n) {
    if (!dst || !src16 || ds4_gpu_tensor_bytes(dst) < n * 4u || ds4_gpu_tensor_bytes(src16) < n * 2u) return 0;
    v41_bind b[] = { V41_T(dst), V41_T(src16), V41_A(n) };
    return v41_launch_1d("kernel_v41_unpack_bf16", b, 3, n);
}
int ds4_gpu_bwd_wcache(int mode) { (void)mode; return 1; }        /* Metal 不落 bf16 暂存, 没有可缓存的解码产物 */
int ds4_gpu_bwd_moe_capture(int on) { (void)on; return 1; }      /* 专家反向照常重算(见 metal_bwd_layer.m) */
