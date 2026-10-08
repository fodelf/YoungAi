/* metal_v41_misc.m — V4.1 小件的 Metal 发射(2026-10-08): 加 / 展开 / 缩放舍入 / argmax / engram 行 / markov 取行·行加·偏置缓存 /
 * 环取行 / hc 均值 / 草稿态低秩件。契约 ds4_gpu_v41.h + ds4_gpu_bwd.h(draft_amp_apply / rows_gather); 核在 metal/v41_dense.metal。 */
#import "metal_v41.h"

int ds4_gpu_v41_add_tensor(ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n) {
    if (!a || !b) return 0;
    v41_bind bd[] = { V41_T(a), V41_T(b), V41_A(n) };
    return v41_launch_1d("kernel_v41_add", bd, 3, n);
}
int ds4_gpu_v41_expand_hc_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!hc || !x) return 0;
    v41_n_args a = { n_embd, n_hc, 0, 0, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(hc), V41_T(x) };
    return v41_launch("kernel_v41_expand_hc", bd, 3, MTLSizeMake((n_embd + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_scale_round_tensor(ds4_gpu_tensor *x, uint64_t n, float s) {
    if (!x || ds4_gpu_tensor_bytes(x) < n * 4) return 0;
    v41_bind bd[] = { V41_T(x), V41_A(n), V41_A(s) };
    return v41_launch_1d("kernel_v41_scale_round", bd, 3, n);
}
int ds4_gpu_v41_argmax_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *logits, uint32_t row, uint32_t n_vocab) {
    if (!idx || !logits || ds4_gpu_tensor_bytes(idx) < 4 || ds4_gpu_tensor_bytes(logits) < ((uint64_t)row + 1) * n_vocab * 4) return 0;
    v41_bind bd[] = { V41_T(idx), V41_TO(logits, (uint64_t)row * n_vocab * 4), V41_A(n_vocab) };
    return v41_launch("kernel_v41_argmax", bd, 3, MTLSizeMake(1, 1, 1), MTLSizeMake(1024, 1, 1));
}
int ds4_gpu_v41_engram_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *raw, uint32_t n_rows, uint32_t head_dim) {
    if (!out || !raw || (head_dim % 32u)) return 0;
    const uint64_t n = (uint64_t)n_rows * head_dim;
    if (ds4_gpu_tensor_bytes(out) < n * 4 || ds4_gpu_tensor_bytes(raw) < (uint64_t)n_rows * (head_dim + head_dim / 32u)) return 0;
    v41_n_args a = { n_rows, head_dim, 0, 0, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(out), V41_T(raw) };
    return v41_launch_1d("kernel_v41_engram_rows", bd, 3, n);
}
int ds4_gpu_v41_row_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t tab_offset, uint64_t n_rows, uint32_t dim,
                                  uint32_t elem_bytes, const ds4_gpu_tensor *ids, uint32_t which, uint32_t out_row) {
    if (!out || !ids || (elem_bytes != 2u && elem_bytes != 4u)) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> tb = v41_model_buf(model_map, model_size, tab_offset, n_rows * dim * elem_bytes, &inner, "markov embed");
    if (!tb) return 0;
    v41_n_args a = { which, dim, out_row, elem_bytes == 4u ? 1u : 0u, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(out), V41_B(tb, 0), V41_T(ids), V41_A(inner) };
    return v41_launch_1d("kernel_v41_row_gather", bd, 5, dim);
}
int ds4_gpu_draft_rows_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t tab_offset, uint64_t n_rows_tab,
                                     uint32_t dim, uint32_t elem_bytes, const ds4_gpu_tensor *ids, uint32_t n) {
    if (!out || !ids || !n || (elem_bytes != 2u && elem_bytes != 4u)) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> tb = v41_model_buf(model_map, model_size, tab_offset, n_rows_tab * dim * elem_bytes, &inner, "dk markov embed");
    if (!tb) return 0;
    v41_n_args a = { 0, dim, 0, elem_bytes == 4u ? 1u : 0u, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(out), V41_B(tb, 0), V41_T(ids), V41_A(inner), V41_A(n_rows_tab) };
    return v41_launch("kernel_v41_rows_gather", bd, 6, MTLSizeMake((dim + 255u) / 256u, n, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_row_add_tensor(ds4_gpu_tensor *dst, uint64_t dst_row, const ds4_gpu_tensor *src, uint64_t n) {
    if (!dst || !src) return 0;
    v41_bind bd[] = { V41_TO(dst, dst_row * n * 4), V41_T(src), V41_A(n) };
    return v41_launch_1d("kernel_v41_add", bd, 3, n);
}
int ds4_gpu_v41_mkcache_lookup_tensor(ds4_gpu_tensor *hit, ds4_gpu_tensor *cache_ids, ds4_gpu_tensor *next, const ds4_gpu_tensor *ids, uint32_t which, uint32_t n_slots) {
    if (!hit || !cache_ids || !next || !ids || n_slots == 0u || n_slots > 1024u) return 0;
    v41_n_args a = { which, n_slots, 0, 0, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(hit), V41_T(cache_ids), V41_T(next), V41_T(ids) };
    return v41_launch("kernel_v41_mkcache_lookup", bd, 5, MTLSizeMake(1, 1, 1), MTLSizeMake(n_slots, 1, 1));
}
int ds4_gpu_v41_mkcache_add_tensor(ds4_gpu_tensor *logits, uint64_t row, const ds4_gpu_tensor *bias, ds4_gpu_tensor *cache, const ds4_gpu_tensor *hit, uint64_t n) {
    if (!logits || !bias || !cache || !hit) return 0;
    v41_bind bd[] = { V41_TO(logits, row * n * 4), V41_T(bias), V41_T(cache), V41_T(hit), V41_A(n) };
    return v41_launch_1d("kernel_v41_mkcache_add", bd, 5, n);
}
int ds4_gpu_v41_ring_rows_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *ring, uint32_t row_floats, uint32_t cap, uint32_t first_row, uint32_t count,
                                 const ds4_gpu_tensor *firstd) {
    if (!dst || !ring || !count || !cap || count > cap) return 0;
    if (ds4_gpu_tensor_bytes(dst) < (uint64_t)count * row_floats * 4 || ds4_gpu_tensor_bytes(ring) < (uint64_t)cap * row_floats * 4) return 0;
    v41_n_args a = { row_floats, cap, first_row, firstd ? 1u : 0u, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(dst), V41_T(ring), V41_T(firstd) };
    return v41_launch("kernel_v41_ring_rows", bd, 4, MTLSizeMake(count, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_hc_mean_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows, uint32_t src_row0, uint32_t slot,
                               uint32_t n_slot, uint32_t dst_pos0, uint32_t cap, const ds4_gpu_tensor *posd) {
    if (!out || !hc || !n_rows || !cap || n_rows > cap) return 0;
    if (ds4_gpu_tensor_bytes(out) < (uint64_t)cap * n_slot * n_embd * 4) return 0;
    v41_hcmean_args a = { n_embd, n_hc, n_rows, src_row0, slot, n_slot, dst_pos0, cap, posd ? 1u : 0u, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(out), V41_T(hc), V41_T(posd) };
    return v41_launch_1d("kernel_v41_hc_mean", bd, 4, (uint64_t)n_rows * n_embd);
}
/* 草稿态小批低秩件: 两发小核(n ≤ 8), 大批转 sgemm 版 */
int ds4_gpu_draft_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B, ds4_gpu_tensor *T,
                                   uint32_t n_tok, uint32_t D, uint32_t K) {
    if (!y || !x || !A || !B || !T || n_tok == 0 || K == 0) return 0;
    if (n_tok > DS4_V41_GEMV_MAX_TOK) return ds4_gpu_v41_amp_apply_tensor(y, x, A, B, T, n_tok, D, K);
    const uint32_t nw = n_tok * K;
    v41_n_args a = { D, K, nw, 0, 0, 0, 0, 0 };
    v41_bind b1[] = { V41_A(a), V41_T(T), V41_T(x), V41_T(B) };
    if (!v41_launch("kernel_v41_lowrank_t", b1, 4, MTLSizeMake((nw + 7u) / 8u, 1, 1), MTLSizeMake(256, 1, 1))) return 0;
    v41_bind b2[] = { V41_A(a), V41_T(y), V41_T(T), V41_T(A) };
    return v41_launch("kernel_v41_lowrank_y", b2, 4, MTLSizeMake((D + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
}
/* markov 偏置表: bias[n][V] = e[n][R]·Wᵀ(W 设备 f32 [V][R]); 反向 gW += gᵀ·e, ge = g·W */
int ds4_gpu_draft_bias_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *e, const ds4_gpu_tensor *W, uint32_t n, uint32_t R, uint32_t V) {
    if (!out || !e || !W || !n || !R || !V) return 0;
    if (n <= DS4_V41_GEMV_MAX_TOK && (R % 8u) == 0u) return v41_f32_gemv_dev(W, R, V, e, out, n, "dk bias gemv");
    return v41_sgemm(e, R, 0, W, R, 1, out, V, n, V, R, 1.0f, 0.0f, "dk bias e·Wᵀ");
}
int ds4_gpu_draft_bias_bwd_tensor(ds4_gpu_tensor *gW, ds4_gpu_tensor *ge, const ds4_gpu_tensor *g, const ds4_gpu_tensor *e, const ds4_gpu_tensor *W,
                                  uint32_t n, uint32_t R, uint32_t V) {
    if (!gW || !ge || !g || !e || !W || !n || !R || !V) return 0;
    /* gW[V][R] += gᵀ[V][n]·e[n][R]: A = g 存 [n][V] ⇒ transA; B = e [n][R] 不转 */
    if (!v41_sgemm(g, V, 1, e, R, 0, gW, R, V, R, n, 1.0f, 1.0f, "dk bias gW+=gᵀe")) return 0;
    /* ge[n][R] = g[n][V]·W[V][R] */
    return v41_sgemm(g, V, 0, W, R, 0, ge, R, n, R, V, 1.0f, 0.0f, "dk bias ge=gW");
}
int ds4_gpu_draft_rows_dev_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tab, const ds4_gpu_tensor *ids, uint32_t ids_off, uint32_t n, uint32_t R, uint32_t Vm) {
    if (!out || !tab || !ids || !n || !R) return 0;
    v41_n_args a = { ids_off, R, Vm, 0, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(out), V41_T(tab), V41_T(ids) };
    return v41_launch("kernel_v41_rows_dev", bd, 4, MTLSizeMake((R + 255u) / 256u, n, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_draft_rows_scatter_tensor(ds4_gpu_tensor *gtab, const ds4_gpu_tensor *g, const ds4_gpu_tensor *ids, uint32_t n, uint32_t R, uint32_t Vm) {
    if (!gtab || !g || !ids || !n || !R) return 0;
    v41_n_args a = { 0, R, Vm, 0, 0, 0, 0, 0 };
    v41_bind bd[] = { V41_A(a), V41_T(gtab), V41_T(g), V41_T(ids) };
    return v41_launch("kernel_v41_rows_scatter", bd, 4, MTLSizeMake((R + 255u) / 256u, n, 1), MTLSizeMake(256, 1, 1));
}
