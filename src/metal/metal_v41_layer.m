/* metal_v41_layer.m — V4.1 层内小核的 Metal 发射(2026-10-08): RMSNorm / mHC 四件 / engram 门 / 路由(含路由偏置侧车) / SwiGLU /
 * 压缩器 / RoPE / 激活量化 / KV 打包 / SWA 窗口环。契约 ds4_gpu_v41.h; 核在 metal/v41_layer.metal。 */
#import "metal_v41.h"

static v41_scratch g_v41_inv;   /* hc_mix 的逐行 1/rms */

int ds4_gpu_v41_rms_norm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                uint32_t dim, uint32_t n_tok, float eps) {
    if (!out || !x || n_tok == 0) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> wb = v41_model_buf(model_map, model_size, weight_offset, (uint64_t)dim * 4, &inner, "v41 norm w");
    if (!wb) return 0;
    v41_n_args a = { dim, 0, 0, 0, eps, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(out), V41_T(x), V41_B(wb, 0), V41_A(inner) };
    return v41_launch("kernel_v41_rms_norm", b, 5, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(256, 1, 1));
}
/* hc_mix 三发: 逐行 1/rms → f32 GEMV/GEMM → 乘回去(CUDA 合一核的三发等价路; 累加序不同, 门是 NLL) */
int ds4_gpu_v41_hc_mix_tensor(ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size, uint64_t fn_offset,
                              uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!mix || !hc || n_tok == 0 || ds4_gpu_tensor_bytes(hc) < (uint64_t)n_tok * dim * 4 || ds4_gpu_tensor_bytes(mix) < (uint64_t)n_tok * mix_hc * 4) return 0;
    id<MTLBuffer> inv = v41_grow(&g_v41_inv, (uint64_t)n_tok * 4, "v41 hc inv");
    if (!inv) return 0;
    v41_n_args a = { dim, 0, 0, 0, eps, 0, 0, 0 };
    v41_bind b1[] = { V41_A(a), V41_B(inv, 0), V41_T(hc) };
    if (!v41_launch("kernel_v41_row_rsqrt", b1, 3, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(256, 1, 1))) return 0;
    if (!ds4_gpu_v41_matmul_f32_tensor(mix, model_map, model_size, fn_offset, dim, mix_hc, hc, n_tok)) return 0;
    const uint64_t n = (uint64_t)n_tok * mix_hc;
    v41_bind b2[] = { V41_T(mix), V41_B(inv, 0), V41_A(mix_hc), V41_A(n) };
    return v41_launch_1d("kernel_v41_scale_rows", b2, 4, n);
}
int ds4_gpu_v41_hc_split_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb, const ds4_gpu_tensor *mix, const void *model_map,
                                uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t iters, float eps, uint32_t n_tok) {
    if (!pre || !post || !comb || !mix || n_hc > 8u || n_tok == 0) return 0;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t so = 0, bo = 0;
    id<MTLBuffer> sb = v41_model_buf(model_map, model_size, scale_offset, 12, &so, "v41 hc scale");
    id<MTLBuffer> bb = v41_model_buf(model_map, model_size, base_offset, (uint64_t)mix_hc * 4, &bo, "v41 hc base");
    if (!sb || !bb) return 0;
    v41_n_args a = { n_hc, iters, 0, 0, eps, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(pre), V41_T(post), V41_T(comb), V41_T(mix), V41_B(sb, (NSUInteger)so), V41_B(bb, (NSUInteger)bo) };
    return v41_launch("kernel_v41_hc_split", b, 7, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(64, 1, 1));
}
int ds4_gpu_v41_hc_pre_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out || !hc || !pre || n_tok == 0) return 0;
    v41_n_args a = { n_embd, n_hc, 0, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(out), V41_T(hc), V41_T(pre) };
    return v41_launch("kernel_v41_hc_pre", b, 4, MTLSizeMake((n_embd + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_hc_fused_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb, ds4_gpu_tensor *x, ds4_gpu_tensor *xn, const ds4_gpu_tensor *mix,
                                const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre_in, const void *model_map, uint64_t model_size, uint64_t scale_offset,
                                uint64_t base_offset, uint64_t norm_offset, uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps, uint32_t n_tok) {
    if (!pre || !post || !comb || !x || !xn || !mix || !hc || !pre_in || n_hc > 5u || n_tok == 0) return 0;   /* n_hc² 要塞进一个 simdgroup */
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t so = 0, bo = 0, no = 0;
    id<MTLBuffer> sb = v41_model_buf(model_map, model_size, scale_offset, 12, &so, "v41 hc scale");
    id<MTLBuffer> bb = v41_model_buf(model_map, model_size, base_offset, (uint64_t)mix_hc * 4, &bo, "v41 hc base");
    id<MTLBuffer> nb = v41_model_buf(model_map, model_size, norm_offset, (uint64_t)n_embd * 4, &no, "v41 norm w");
    if (!sb || !bb || !nb) return 0;
    v41_n_args a = { n_embd, n_hc, iters, 0, hc_eps, norm_eps, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(pre), V41_T(post), V41_T(comb), V41_T(x), V41_T(xn), V41_T(mix), V41_T(hc), V41_T(pre_in),
                     V41_B(sb, (NSUInteger)so), V41_B(bb, (NSUInteger)bo), V41_B(nb, (NSUInteger)no) };
    return v41_launch("kernel_v41_hc_fused", b, 12, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(1024, 1, 1));
}
int ds4_gpu_v41_hc_post_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                               uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out_hc || !y || !res || !post || !comb || n_tok == 0) return 0;
    if (ds4_gpu_tensor_buffer(out_hc) == ds4_gpu_tensor_buffer(res) && ds4_gpu_tensor_offset(out_hc) == ds4_gpu_tensor_offset(res)) return 0;   /* 不许原地 */
    v41_n_args a = { n_embd, n_hc, 0, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(out_hc), V41_T(y), V41_T(res), V41_T(post), V41_T(comb) };
    return v41_launch("kernel_v41_hc_post", b, 6, MTLSizeMake((n_embd + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_engram_gate_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size, uint64_t q_w_offset, uint64_t k_w_offset,
                                   uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    if (!hc || !kv || n_tok == 0) return 0;
    uint64_t qo = 0, ko = 0;
    id<MTLBuffer> qb = v41_model_buf(model_map, model_size, q_w_offset, (uint64_t)n_hc * n_embd * 4, &qo, "v41 engram q");
    id<MTLBuffer> kb = v41_model_buf(model_map, model_size, k_w_offset, (uint64_t)n_hc * n_embd * 4, &ko, "v41 engram k");
    if (!qb || !kb) return 0;
    v41_n_args a = { n_embd, n_hc, 0, 0, eps, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(hc), V41_T(kv), V41_B(qb, (NSUInteger)qo), V41_B(kb, (NSUInteger)ko) };
    return v41_launch("kernel_v41_engram_gate", b, 5, MTLSizeMake(n_hc, n_tok, 1), MTLSizeMake(256, 1, 1));
}

/* 路由偏置侧车: [exp_probs_b 文件偏移] → 设备 (盘上 bias + Δb)[E]; 按偏移认层 */
static struct { uint64_t off; __strong id<MTLBuffer> dev; uint32_t n; } g_v41_rb[64];
static uint32_t g_v41_rb_n = 0;
int ds4_gpu_v41_set_rb_override(const void *model_map, uint64_t model_size, uint64_t bias_offset, const float *host_delta, uint32_t n_expert) {
    if (!host_delta) {
        for (uint32_t i = 0; i < g_v41_rb_n;) {
            if (bias_offset == 0 || g_v41_rb[i].off == bias_offset) { g_v41_rb[i].dev = nil; g_v41_rb[i] = g_v41_rb[--g_v41_rb_n]; g_v41_rb[g_v41_rb_n].dev = nil; }
            else i++;
        }
        return 1;
    }
    if (!model_map || !n_expert || bias_offset > model_size || (uint64_t)n_expert * 4 > model_size - bias_offset) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const float *base = (const float *)((const uint8_t *)model_map + bias_offset);   /* 盘上 bias 直接从主机映射读 */
    uint32_t i = 0;
    for (; i < g_v41_rb_n; i++) if (g_v41_rb[i].off == bias_offset) break;
    if (i == g_v41_rb_n) {
        if (g_v41_rb_n >= 64u) return 0;
        g_v41_rb[i].dev = [g_device newBufferWithLength:(NSUInteger)n_expert * 4 options:MTLResourceStorageModeShared];
        if (!g_v41_rb[i].dev) return 0;
        g_v41_rb[i].off = bias_offset; g_v41_rb[i].n = n_expert; g_v41_rb_n++;
    } else if (g_v41_rb[i].n != n_expert) return 0;
    float *d = (float *)[g_v41_rb[i].dev contents];
    for (uint32_t e = 0; e < n_expert; e++) d[e] = base[e] + host_delta[e];
    return 1;
}
int ds4_gpu_v41_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, const void *model_map, uint64_t model_size,
                              uint64_t bias_offset, uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    if (!selected || !weights || !logits || topk > 16u || n_expert > 32u * 12u || n_tok == 0) return 0;
    uint64_t bo = 0;
    id<MTLBuffer> bb = v41_model_buf(model_map, model_size, bias_offset, (uint64_t)n_expert * 4, &bo, "v41 gate bias");
    if (!bb) return 0;
    for (uint32_t i = 0; i < g_v41_rb_n; i++) if (g_v41_rb[i].off == bias_offset && g_v41_rb[i].n == n_expert) { bb = g_v41_rb[i].dev; bo = 0; break; }
    v41_router_args a = { n_tok, n_expert, topk, 0, route_scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(selected), V41_T(weights), V41_T(logits), V41_B(bb, (NSUInteger)bo) };
    return v41_launch("kernel_v41_router", b, 5, MTLSizeMake((n_tok + 7u) / 8u, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_swiglu_tensor(ds4_gpu_tensor *h, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n_tok, uint32_t mid, float limit) {
    if (!h || !gate || !up) return 0;
    const uint64_t n = (uint64_t)n_tok * mid;
    v41_bind b[] = { V41_T(h), V41_T(gate), V41_T(up), V41_A(n), V41_A(limit) };
    return v41_launch_1d("kernel_v41_swiglu", b, 5, n);
}
int ds4_gpu_v41_compress_pool_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ratio, uint32_t dim) {
    if (!out || !kv || !score || ratio == 0) return 0;
    const uint32_t ng = n_tok / ratio;
    if (ng == 0) return 1;
    v41_n_args a = { ratio, dim, 0, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(out), V41_T(kv), V41_T(score) };
    return v41_launch("kernel_v41_compress_pool", b, 4, MTLSizeMake(ng, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_compress_step_n_tensor(ds4_gpu_tensor *pooled, ds4_gpu_tensor *posg, ds4_gpu_tensor *cpre_kv, ds4_gpu_tensor *cpre_sc, ds4_gpu_tensor *snap_kv,
                                       ds4_gpu_tensor *snap_sc, const ds4_gpu_tensor *ckv, const ds4_gpu_tensor *csc, const ds4_gpu_tensor *posd,
                                       uint32_t ratio, uint32_t dim, uint32_t n) {
    if (!pooled || !posg || !cpre_kv || !cpre_sc || !ckv || !csc || !posd || ratio < 2u || n == 0u || n > 8u) return 0;
    const uint32_t ngmax = (ratio - 1u + n) / ratio;
    if (ds4_gpu_tensor_bytes(cpre_kv) < (uint64_t)ratio * dim * 4 || ds4_gpu_tensor_bytes(cpre_sc) < (uint64_t)ratio * dim * 4) return 0;
    if (ds4_gpu_tensor_bytes(pooled) < (uint64_t)ngmax * dim * 4 || ds4_gpu_tensor_bytes(posg) < (uint64_t)ngmax * 4 || ds4_gpu_tensor_bytes(ckv) < (uint64_t)n * dim * 4) return 0;
    if ((snap_kv != NULL) != (snap_sc != NULL)) return 0;
    if (snap_kv && (ds4_gpu_tensor_bytes(snap_kv) < (uint64_t)(ratio - 1u + n) * dim * 4 || ds4_gpu_tensor_bytes(snap_sc) < (uint64_t)(ratio - 1u + n) * dim * 4)) return 0;
    v41_cstep_args a = { ratio, dim, n, snap_kv ? 1u : 0u };
    v41_bind b[] = { V41_A(a), V41_T(pooled), V41_T(posg), V41_T(cpre_kv), V41_T(cpre_sc), V41_T(snap_kv), V41_T(snap_sc), V41_T(ckv), V41_T(csc), V41_T(posd) };
    return v41_launch("kernel_v41_compress_step_n", b, 10, MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1));
}
static int v41_rope(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, float theta, uint32_t osl,
                    float factor, float beta_fast, float beta_slow, bool inverse, int do_round) {
    if (!x || !pos || (n_rot & 1u) || n_rot > 128u || n_tok == 0) return 0;
    v41_rope_args a = { n_head, head_dim, n_rot, osl, inverse ? 1u : 0u, (uint32_t)do_round, theta, factor, beta_fast, beta_slow };
    v41_bind b[] = { V41_A(a), V41_T(x), V41_T(pos) };
    return v41_launch("kernel_v41_rope", b, 3, MTLSizeMake(n_head, n_tok, 1), MTLSizeMake(64, 1, 1));
}
int ds4_gpu_v41_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, float theta,
                            uint32_t original_seq_len, float factor, float beta_fast, float beta_slow, bool inverse) {
    return v41_rope(x, pos, n_tok, n_head, head_dim, n_rot, theta, original_seq_len, factor, beta_fast, beta_slow, inverse, 1);
}
int ds4_gpu_bwd_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, float theta,
                            uint32_t original_seq_len, float factor, float beta_fast, float beta_slow, bool inverse) {
    return v41_rope(x, pos, n_tok, n_head, head_dim, n_rot, theta, original_seq_len, factor, beta_fast, beta_slow, inverse, 0);
}
static int v41_act_quant(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, uint32_t mode) {
    if (!x || block == 0 || block > 32u || (dim % block) || n_rows == 0) return 0;
    const uint32_t nb_row = dim / block;
    v41_n_args a = { dim, block, mode, nb_row, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(x) };
    return v41_launch("kernel_v41_act_quant", b, 2, MTLSizeMake(((uint64_t)n_rows * nb_row + 7u) / 8u, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_act_quant_fp8_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block) { return v41_act_quant(x, n_rows, dim, block, 0u); }
int ds4_gpu_v41_act_quant_fp4_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, bool e4m3_scale) { return v41_act_quant(x, n_rows, dim, block, e4m3_scale ? 2u : 1u); }
static int v41_kv_pack(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows, const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash,
                       uint32_t nbatch, uint32_t blk, uint32_t row_bytes, uint32_t nib_bytes, uint32_t mode) {
    if (!cache || !rows || !n_rows) return 0;
    if (!posd) { if (ds4_gpu_tensor_bytes(cache) < (uint64_t)(g0 + n_rows) * row_bytes) return 0; }
    else { if (ratio == 0u || nbatch == 0u || n_rows > (ratio - 1u + nbatch) / ratio || ds4_gpu_tensor_bytes(cache) < ((uint64_t)g_trash + 1u) * row_bytes) return 0; }
    const uint32_t dim = nib_bytes * 2u, nb = dim / blk;
    v41_kvpack_args a = { g0, dim, blk, row_bytes, nib_bytes, nb, mode, posd ? 1u : 0u, ratio, g_trash, nbatch, 0 };
    v41_bind b[] = { V41_A(a), V41_T(cache), V41_T(rows), V41_T(posd) };
    return v41_launch("kernel_v41_kv_pack", b, 4, MTLSizeMake(((uint64_t)n_rows * nb + 7u) / 8u, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_ckv_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows, const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch) {
    return v41_kv_pack(cache, g0, rows, n_rows, posd, ratio, g_trash, nbatch, DS4_V41_CKV_BLK, DS4_V41_CKV_BYTES, DS4_V41_CKV_NIB, 0u);
}
int ds4_gpu_v41_idxk_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows, const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch) {
    return v41_kv_pack(cache, g0, rows, n_rows, posd, ratio, g_trash, nbatch, DS4_V41_IDXK_BLK, DS4_V41_IDXK_BYTES, DS4_V41_IDXK_NIB, 1u);
}
int ds4_gpu_v41_win_commit_tensor(ds4_gpu_tensor *win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim, const ds4_gpu_tensor *posd) {
    if (!win || !window || !n) return 0;
    if (ds4_gpu_tensor_bytes(win) < (uint64_t)(window + n) * head_dim * 4) return 0;
    if (posd && n > 8u) return 0;
    const uint32_t i0 = n > window ? n - window : 0u, rows = n - i0;
    v41_ring_args a = { pos0, i0, window, head_dim, 0, posd ? 1u : 0u, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(win), V41_T(posd) };
    return v41_launch("kernel_v41_win_commit", b, 3, MTLSizeMake(rows, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_win_ring_snap_tensor(ds4_gpu_tensor *win, ds4_gpu_tensor *snap, uint32_t pos0, uint32_t i0, uint32_t n, uint32_t window, uint32_t head_dim, int back,
                                     const ds4_gpu_tensor *posd) {
    if (!win || !snap || !window || n <= i0) return 1;
    if (ds4_gpu_tensor_bytes(snap) < (uint64_t)n * head_dim * 4) return 0;
    v41_ring_args a = { pos0, i0, window, head_dim, back ? 1u : 0u, posd ? 1u : 0u, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(win), V41_T(snap), V41_T(posd) };
    return v41_launch("kernel_v41_win_snap", b, 4, MTLSizeMake(n - i0, 1, 1), MTLSizeMake(256, 1, 1));
}
