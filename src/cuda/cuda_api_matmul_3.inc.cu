/* cuda_api_matmul_3.inc.cu — 压缩器攒批投影的发射(2026-09-05, 核在 cuda_compressor_kernels)。
 * ds4_gpu_matmul_f16_pair_rows_tensor: n 行 x 一次过双矩阵, 每行数值与单 token pair 核逐字相同。
 * ds4_gpu_compressor_ring_push_tensor: 把当前 token 的 x 行推进层环(emit 时取回攒批)。
 */
int ds4_gpu_matmul_f16_pair_rows_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        (in_dim & 7u) != 0 || n_tok > 4096u ||
        weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out0->bytes < n_tok * out_dim * sizeof(float) ||
        out1->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    unsigned pgrid = (unsigned)(2u * out_dim);   /* 与单 token pair 核同款: 2*out 行, 常驻块行循环 */
    if (pgrid > ds4_grid_cap()) pgrid = ds4_grid_cap();
    ds4_launch_pdl(matmul_f16_pair_rowblock_mtok_kernel, pgrid, 256, 0, g_cur_stream, 
            (float *)out0->ptr, (float *)out1->ptr, w0, w1, (const float *)x->ptr,
            in_dim, out_dim, (uint32_t)n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_rowblock_mtok launch");
}

int ds4_gpu_compressor_ring_push_tensor(
        ds4_gpu_tensor *ring,
        uint32_t row,
        const ds4_gpu_tensor *x,
        uint32_t n) {
    if (!ring || !x || n == 0 || (n & 3u) != 0 ||
        x->bytes < (uint64_t)n * sizeof(float) ||
        ring->bytes < ((uint64_t)row + 1u) * n * sizeof(float)) {
        return 0;
    }
    const uint32_t n4 = n >> 2;
    ds4_launch_pdl(comp_ring_push_kernel, (n4 + 255u) / 256u, 256, 0, g_cur_stream, 
            (float4 *)((float *)ring->ptr + (uint64_t)row * n), (const float4 *)x->ptr, n4);
    return cuda_ok(cudaGetLastError(), "comp ring push launch");
}
