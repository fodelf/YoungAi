/* metal_compressor_batch.m — 压缩器攒批契约的 Metal 侧(2026-09-05)。
 *
 * CUDA 侧(cuda_compressor_kernels / cuda_api_matmul_3)用多行核把权重读一遍算 8 个 token;
 * Metal 这里按契约语义逐 token 调单 token 入口: 每行数值与单 token 路逐字相同(同一个核),
 * 只是权重多读几遍 —— Mac 不是解码速度战场(CLAUDE.md: Metal 主后端但速度战役在 spark),
 * 契约先对齐, 攒批核后补。 */
#import "metal_internal.h"

int ds4_gpu_compressor_update_batch_tensor(
        const ds4_gpu_tensor *kv_cur, const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, ds4_gpu_tensor *comp_cache,
        const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type,
        uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio,
        uint32_t pos0, uint32_t n_tokens, uint32_t comp_row, uint32_t n_rot, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float rms_eps) {
    if (!kv_cur || !sc_cur || head_dim == 0 || ratio == 0 || n_tokens == 0 || n_tokens > ratio) return 0;
    const uint64_t row_bytes = (uint64_t)(ratio == 4u ? 2u : 1u) * head_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(kv_cur) < row_bytes * n_tokens || ds4_gpu_tensor_bytes(sc_cur) < row_bytes * n_tokens)
        return 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *kv_t = ds4_gpu_tensor_view(kv_cur, (uint64_t)t * row_bytes, row_bytes);
        ds4_gpu_tensor *sc_t = ds4_gpu_tensor_view(sc_cur, (uint64_t)t * row_bytes, row_bytes);
        int ok = kv_t && sc_t &&
                 ds4_gpu_compressor_update_tensor(kv_t, sc_t, state_kv, state_score, comp_cache,
                                                  model_map, model_size, ape_offset, ape_type,
                                                  norm_offset, norm_type, head_dim, ratio, pos0 + t,
                                                  comp_row, n_rot, n_ctx_orig, freq_base, freq_scale,
                                                  ext_factor, attn_factor, beta_fast, beta_slow, rms_eps);
        ds4_gpu_tensor_free(kv_t);
        ds4_gpu_tensor_free(sc_t);
        if (!ok) return 0;
    }
    return 1;
}

int ds4_gpu_matmul_f16_pair_rows_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!out_a || !out_b || !x || n_tok == 0 || in_dim == 0 || out_dim == 0) return 0;
    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t o_bytes = out_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(x) < x_bytes * n_tok ||
        ds4_gpu_tensor_bytes(out_a) < o_bytes * n_tok || ds4_gpu_tensor_bytes(out_b) < o_bytes * n_tok)
        return 0;
    for (uint64_t t = 0; t < n_tok; t++) {
        ds4_gpu_tensor *xt = ds4_gpu_tensor_view(x, t * x_bytes, x_bytes);
        ds4_gpu_tensor *at = ds4_gpu_tensor_view(out_a, t * o_bytes, o_bytes);
        ds4_gpu_tensor *bt = ds4_gpu_tensor_view(out_b, t * o_bytes, o_bytes);
        int ok = xt && at && bt &&
                 ds4_gpu_matmul_f16_pair_tensor(at, bt, model_map, model_size, weight_a_offset,
                                                weight_b_offset, in_dim, out_dim, xt, 1);
        ds4_gpu_tensor_free(xt);
        ds4_gpu_tensor_free(at);
        ds4_gpu_tensor_free(bt);
        if (!ok) return 0;
    }
    return 1;
}

int ds4_gpu_compressor_ring_push_tensor(ds4_gpu_tensor *ring, uint32_t row,
                                        const ds4_gpu_tensor *x, uint32_t n) {
    if (!ring || !x || n == 0) return 0;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    return ds4_gpu_tensor_copy(ring, (uint64_t)row * bytes, x, 0, bytes);
}
