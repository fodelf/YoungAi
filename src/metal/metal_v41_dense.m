/* metal_v41_dense.m — V4.1 稠密线性层的 Metal 发射(2026-10-08): fp4x32 / q4_K / bf16 / f32 / fp8blk 的 GEMV(n ≤ 8)与预填 GEMM、
 * 分组块对角(wo_a)、嵌入取行、出口头列平方和、f32 GEMM(放大器)。契约 ds4_gpu_v41.h; 核在 metal/v41_dense.metal。
 * 与 CUDA 的差别只有发法: CUDA 预填把权重解成 bf16 再 cuBLAS, 这里的 wgemm 边解边乘(值同一张表, 累加序不同 ⇒ 门是 NLL 不是 cmp)。 */
#import "metal_v41.h"

/* 一个 [rows][cols] 矩阵在盘上占多少字节(fp8blk 另有缩放平面, 由调用方加) */
static uint64_t v41_wbytes(uint32_t wtype, uint64_t rows, uint64_t cols) {
    switch (wtype) {
        case V41_WT_FP4X32: return rows * (cols / 32u) * 17u;
        case V41_WT_Q4K:    return rows * (cols / 256u) * 144u;
        case V41_WT_BF16:   return rows * cols * 2u;
        case V41_WT_F32:    return rows * cols * 4u;
        default:            return rows * cols;   /* fp8 e4m3 平面 */
    }
}
static int v41_wtype_ok(uint32_t wtype, uint64_t cols, const char *what) {
    const uint64_t need = wtype == V41_WT_Q4K ? 256u : (wtype == V41_WT_FP4X32 || wtype == V41_WT_FP8BLK ? 32u : 8u);
    if (cols % need) { fprintf(stderr, "ds4: [v41-metal] %s: 列数 %llu 不是 %llu 的倍数\n", what, (unsigned long long)cols, (unsigned long long)need); return 0; }
    return 1;
}

int v41_gemv(const void *model_map, uint64_t model_size, uint32_t wtype, uint64_t off, uint64_t in_dim, uint64_t out_dim,
             const ds4_gpu_tensor *x, uint32_t x_stride, ds4_gpu_tensor *out, uint32_t out_stride, uint32_t n_tok,
             uint32_t n_groups, uint64_t w_gstride, uint32_t x_gstride, uint32_t out_gstride, int round_out, const ds4_gpu_tensor *skip, const char *what) {
    if (!x || !out || n_tok == 0 || n_tok > DS4_V41_GEMV_MAX_TOK || n_groups == 0 || !v41_wtype_ok(wtype, in_dim, what)) return 0;
    const uint64_t wg = v41_wbytes(wtype, out_dim, in_dim);
    uint64_t total = wtype == V41_WT_FP8BLK ? wg * n_groups + (uint64_t)n_groups * ((out_dim + 31u) / 32u) * ((in_dim + 31u) / 32u) : wg * n_groups;
    if (n_groups > 1u && wtype != V41_WT_FP8BLK) total = w_gstride * (n_groups - 1u) + wg;
    uint64_t inner = 0;
    id<MTLBuffer> wb = v41_model_buf(model_map, model_size, off, total, &inner, what);
    if (!wb) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit * n_groups < 8192u) ksplit <<= 1;
    const uint32_t nchunk = (uint32_t)((in_dim + 255u) / 256u);
    while (ksplit > 1u && ksplit > nchunk) ksplit >>= 1;
    v41_gemv_args a;
    memset(&a, 0, sizeof a);
    a.w_off = inner; a.w_gstride = wtype == V41_WT_FP8BLK ? in_dim * out_dim : w_gstride;
    a.sbc = (uint32_t)((in_dim + 31u) / 32u);
    a.sc_off = wtype == V41_WT_FP8BLK ? inner + in_dim * out_dim * n_groups : inner;
    a.sc_gstride = wtype == V41_WT_FP8BLK ? (uint64_t)((out_dim + 31u) / 32u) * a.sbc : 0;
    a.in_dim = (uint32_t)in_dim; a.out_dim = (uint32_t)out_dim; a.x_stride = x_stride; a.out_stride = out_stride; a.ksplit = ksplit;
    a.x_gstride = x_gstride; a.out_gstride = out_gstride; a.n_tok = n_tok; a.round_out = round_out ? 1u : 0u; a.wtype = wtype; a.has_skip = skip ? 1u : 0u;
    char name[48]; snprintf(name, sizeof name, "kernel_v41_gemv_nt%u", n_tok);
    const uint32_t rpb = 8u / ksplit;
    v41_bind b[] = { V41_A(a), V41_B(wb, 0), V41_T(x), V41_T(out), V41_T(skip), V41_B(wb, 0) };
    return v41_launch(name, b, 6, MTLSizeMake((NSUInteger)((out_dim + rpb - 1u) / rpb), n_groups, 1), MTLSizeMake(256, 1, 1));
}

int v41_wgemm(const void *model_map, uint64_t model_size, uint32_t wtype, uint64_t off, uint32_t wnn, uint32_t M, uint32_t N, uint32_t K,
              const ds4_gpu_tensor *x, uint32_t lda, ds4_gpu_tensor *out, uint32_t ldc, uint32_t n_groups, uint64_t w_gstride,
              uint32_t x_gstride, uint32_t out_gstride, int beta, int round_out, const char *what) {
    if (!x || !out || M == 0 || N == 0 || K == 0 || n_groups == 0) return 0;
    const uint32_t wrows = wnn ? K : N, wcols = wnn ? N : K;
    if (!v41_wtype_ok(wtype, wcols, what)) return 0;
    const uint64_t wg = v41_wbytes(wtype, wrows, wcols);
    const uint64_t sbc = (wcols + 31u) / 32u, sbr = (wrows + 31u) / 32u;
    uint64_t total = wtype == V41_WT_FP8BLK ? wg * n_groups + (uint64_t)n_groups * sbr * sbc : (n_groups > 1u ? w_gstride * (n_groups - 1u) + wg : wg);
    uint64_t inner = 0;
    id<MTLBuffer> wb = v41_model_buf(model_map, model_size, off, total, &inner, what);
    if (!wb) return 0;
    v41_wgemm_args a;
    memset(&a, 0, sizeof a);
    a.w_off = inner; a.w_gstride = wtype == V41_WT_FP8BLK ? wg : w_gstride;
    a.sc_off = wtype == V41_WT_FP8BLK ? inner + wg * n_groups : inner; a.sc_gstride = wtype == V41_WT_FP8BLK ? sbr * sbc : 0;
    a.M = M; a.N = N; a.K = K; a.lda = lda; a.ldc = ldc; a.x_gstride = x_gstride; a.out_gstride = out_gstride;
    a.wtype = wtype; a.wnn = wnn; a.wcols = wcols; a.sbc = (uint32_t)sbc; a.beta = beta ? 1u : 0u; a.round_out = round_out ? 1u : 0u;
    v41_bind b[] = { V41_A(a), V41_B(wb, 0), V41_T(x), V41_T(out), V41_B(wb, 0) };
    return v41_launch("kernel_v41_wgemm", b, 5, MTLSizeMake((N + 63u) / 64u, (M + 63u) / 64u, n_groups), MTLSizeMake(128, 1, 1));
}

int v41_sgemm(const ds4_gpu_tensor *A, uint32_t lda, int transA, const ds4_gpu_tensor *B, uint32_t ldb, int transB, ds4_gpu_tensor *C, uint32_t ldc,
              uint32_t M, uint32_t N, uint32_t K, float alpha, float beta, const char *what) {
    (void)what;
    if (!A || !B || !C || M == 0 || N == 0 || K == 0) return 0;
    v41_sgemm_args a;
    memset(&a, 0, sizeof a);
    a.M = M; a.N = N; a.K = K; a.lda = lda; a.ldb = ldb; a.ldc = ldc; a.transA = transA ? 1u : 0u; a.transB = transB ? 1u : 0u; a.alpha = alpha; a.beta = beta;
    v41_bind b[] = { V41_A(a), V41_T(A), V41_T(B), V41_T(C) };
    return v41_launch("kernel_v41_sgemm", b, 4, MTLSizeMake((N + 63u) / 64u, (M + 63u) / 64u, 1), MTLSizeMake(128, 1, 1));
}
/* 设备 f32 权重的 GEMV(权重不在模型映射里, 直接绑张量) */
int v41_f32_gemv_dev(const ds4_gpu_tensor *W, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, ds4_gpu_tensor *out, uint32_t n_tok, const char *what) {
    (void)what;
    if (!W || !x || !out || n_tok == 0 || n_tok > DS4_V41_GEMV_MAX_TOK || (in_dim % 8u)) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit < 8192u) ksplit <<= 1;
    const uint32_t nchunk = (uint32_t)((in_dim + 255u) / 256u);
    while (ksplit > 1u && ksplit > nchunk) ksplit >>= 1;
    v41_gemv_args a;
    memset(&a, 0, sizeof a);
    a.in_dim = (uint32_t)in_dim; a.out_dim = (uint32_t)out_dim; a.x_stride = (uint32_t)in_dim; a.out_stride = (uint32_t)out_dim; a.ksplit = ksplit; a.n_tok = n_tok; a.wtype = V41_WT_F32;
    char name[48]; snprintf(name, sizeof name, "kernel_v41_gemv_nt%u", n_tok);
    const uint32_t rpb = 8u / ksplit;
    v41_bind b[] = { V41_A(a), V41_T(W), V41_T(x), V41_T(out), V41_T(NULL), V41_T(W) };
    return v41_launch(name, b, 6, MTLSizeMake((NSUInteger)((out_dim + rpb - 1u) / rpb), 1, 1), MTLSizeMake(256, 1, 1));
}

/* ---- 契约入口: 按类型分发到 GEMV(n ≤ 8)/GEMM ---- */
static int v41_matmul(uint32_t wtype, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t off, uint64_t in_dim, uint64_t out_dim,
                      const ds4_gpu_tensor *x, uint32_t n_tok, int round_out, const char *what) {
    if (!out || !x || n_tok == 0) return 0;
    if (ds4_gpu_tensor_bytes(x) < (uint64_t)n_tok * in_dim * 4 || ds4_gpu_tensor_bytes(out) < (uint64_t)n_tok * out_dim * 4) return 0;
    if (n_tok <= DS4_V41_GEMV_MAX_TOK)
        return v41_gemv(model_map, model_size, wtype, off, in_dim, out_dim, x, (uint32_t)in_dim, out, (uint32_t)out_dim, n_tok, 1u, 0, 0, 0, round_out, NULL, what);
    return v41_wgemm(model_map, model_size, wtype, off, 0u, n_tok, (uint32_t)out_dim, (uint32_t)in_dim, x, (uint32_t)in_dim, out, (uint32_t)out_dim, 1u, 0, 0, 0, 0, round_out, what);
}
static int v41_grouped(uint32_t wtype, ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t off, uint32_t n_groups, uint64_t group_dim,
                       uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out, const char *what) {
    if (!low || !heads || n_tok == 0 || n_groups == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    if (ds4_gpu_tensor_bytes(heads) < (uint64_t)n_tok * in_all * 4 || ds4_gpu_tensor_bytes(low) < (uint64_t)n_tok * out_all * 4) return 0;
    const uint64_t wg = v41_wbytes(wtype, rank, group_dim);
    if (n_tok <= DS4_V41_GEMV_MAX_TOK)
        return v41_gemv(model_map, model_size, wtype, off, group_dim, rank, heads, (uint32_t)in_all, low, (uint32_t)out_all, n_tok, n_groups, wg,
                        (uint32_t)group_dim, (uint32_t)rank, round_out, NULL, what);
    return v41_wgemm(model_map, model_size, wtype, off, 0u, n_tok, (uint32_t)rank, (uint32_t)group_dim, heads, (uint32_t)in_all, low, (uint32_t)out_all,
                     n_groups, wg, (uint32_t)group_dim, (uint32_t)rank, 0, round_out, what);
}
int ds4_gpu_v41_matmul_fp4x32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    return v41_matmul(V41_WT_FP4X32, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok, round_out, "v41 fp4x32 matmul");
}
int ds4_gpu_v41_grouped_matmul_fp4x32_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_groups,
                                             uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    return v41_grouped(V41_WT_FP4X32, low, model_map, model_size, weight_offset, n_groups, group_dim, rank, heads, n_tok, round_out, "v41 wo_a fp4x32");
}
int ds4_gpu_v41_matmul_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    return v41_matmul(V41_WT_Q4K, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok, round_out, "v41 q4k matmul");
}
int ds4_gpu_v41_grouped_matmul_q4k_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_groups,
                                          uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    return v41_grouped(V41_WT_Q4K, low, model_map, model_size, weight_offset, n_groups, group_dim, rank, heads, n_tok, round_out, "v41 wo_a q4k");
}
int ds4_gpu_v41_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok) {
    return v41_matmul(V41_WT_F32, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok, 0, "v41 f32 matmul");
}
int ds4_gpu_v41_matmul_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                   const ds4_gpu_tensor *x, uint32_t n_tok) {
    return v41_matmul(V41_WT_BF16, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok, 0, "v41 bf16 matmul");
}
int ds4_gpu_v41_matmul_bf16_skip_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                        const ds4_gpu_tensor *x, uint32_t n_tok, const ds4_gpu_tensor *skip) {
    if (!skip || n_tok > DS4_V41_GEMV_MAX_TOK) return 0;
    return v41_gemv(model_map, model_size, V41_WT_BF16, weight_offset, in_dim, out_dim, x, (uint32_t)in_dim, out, (uint32_t)out_dim, n_tok, 1u, 0, 0, 0, 0, skip, "v41 bf16 gemv(skip)");
}
int ds4_gpu_v41_matmul_fp8blk_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok) {
    return v41_matmul(V41_WT_FP8BLK, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok, 0, "v41 fp8blk matmul");
}
int ds4_gpu_v41_matmul_fp8blk_round_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                           const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    return v41_matmul(V41_WT_FP8BLK, out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok, round_out, "v41 mtp fp8 matmul");
}
int ds4_gpu_v41_grouped_matmul_fp8blk_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_groups,
                                             uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    return v41_grouped(V41_WT_FP8BLK, low, model_map, model_size, weight_offset, n_groups, group_dim, rank, heads, n_tok, round_out, "v41 mtp wo_a fp8");
}

/* ---- 嵌入 / 列平方和 / bf16 舍入 ---- */
static int v41_embed(uint32_t wtype, ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size, uint64_t off,
                     uint64_t n_vocab, uint32_t n_tok, uint64_t dim) {
    if (!out || !tokens || n_tok == 0 || !v41_wtype_ok(wtype, dim, "v41 embed")) return 0;
    if (ds4_gpu_tensor_bytes(out) < (uint64_t)n_tok * dim * 4) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> wb = v41_model_buf(model_map, model_size, off, v41_wbytes(wtype, n_vocab, dim), &inner, "v41 embed");
    if (!wb) return 0;
    v41_n_args a = { (uint32_t)n_vocab, (uint32_t)dim, wtype, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_B(wb, 0), V41_T(tokens), V41_T(out), V41_A(inner) };
    return v41_launch("kernel_v41_embed", b, 5, MTLSizeMake((NSUInteger)((dim / 8u + 255u) / 256u), n_tok, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_embed_fp4x32_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                    uint32_t n_vocab, uint32_t n_tok, uint32_t n_embd) {
    return v41_embed(V41_WT_FP4X32, out, tokens, model_map, model_size, weight_offset, n_vocab, n_tok, n_embd);
}
int ds4_gpu_v41_embed_q4k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                 uint64_t n_vocab, uint32_t n_tok, uint64_t dim) {
    return v41_embed(V41_WT_Q4K, out, tokens, model_map, model_size, weight_offset, n_vocab, n_tok, dim);
}
static int v41_colnorm(uint32_t wtype, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t off, uint32_t n_vocab, uint32_t n_embd) {
    if (!out || !v41_wtype_ok(wtype, n_embd, "v41 head colnorm") || ds4_gpu_tensor_bytes(out) < (uint64_t)n_embd * 4) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> wb = v41_model_buf(model_map, model_size, off, v41_wbytes(wtype, n_vocab, n_embd), &inner, "v41 head colnorm");
    if (!wb) return 0;
    v41_n_args a = { n_vocab, n_embd, wtype, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_B(wb, 0), V41_T(out), V41_A(inner) };
    return v41_launch("kernel_v41_colnorm", b, 4, MTLSizeMake(n_embd, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_head_colnorm_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd) {
    return v41_colnorm(V41_WT_FP4X32, out, model_map, model_size, weight_offset, n_vocab, n_embd);
}
int ds4_gpu_v41_head_colnorm_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd) {
    return v41_colnorm(V41_WT_Q4K, out, model_map, model_size, weight_offset, n_vocab, n_embd);
}
int ds4_gpu_v41_round_bf16_tensor(ds4_gpu_tensor *x, uint64_t n) {
    if (!x || ds4_gpu_tensor_bytes(x) < n * 4) return 0;
    v41_bind b[] = { V41_T(x), V41_A(n) };
    return v41_launch_1d("kernel_v41_round_bf16", b, 2, n);
}
/* 反修放大器: y[n][D] += x[n][D]·(B·A), A/B 设备 [K][D] 行主序 ⇒ T[n][K] = x·Bᵀ, y += T·A */
int ds4_gpu_v41_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B, ds4_gpu_tensor *T,
                                 uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!y || !x || !A || !B || !T || n_tok == 0 || K == 0) return 0;
    if (ds4_gpu_tensor_bytes(T) < (uint64_t)n_tok * K * 4 || ds4_gpu_tensor_bytes(y) < (uint64_t)n_tok * D * 4 || ds4_gpu_tensor_bytes(x) < (uint64_t)n_tok * D * 4) return 0;
    if (!v41_sgemm(x, D, 0, B, D, 1, T, K, n_tok, K, D, 1.0f, 0.0f, "v41 amp T=xB")) return 0;
    return v41_sgemm(T, K, 0, A, D, 0, y, D, n_tok, D, K, 1.0f, 1.0f, "v41 amp y+=TA");
}
