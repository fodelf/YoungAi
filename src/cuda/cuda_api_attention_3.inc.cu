/* cuda_api_attention_3.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * attention GPU API(decode/prefill/indexed/output 投影)。
 */
int ds4_gpu_attention_output_low_q4k_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    /* q4_K 版 attn_output_a(与 q8 入口同构): 权重行 = group_dim/256 个 q4_K 块。 */
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) return 0;
    if (group_dim % 256u != 0) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = group_dim / 32u;          /* q8_0 激活块数 */
    const uint64_t kblocks = group_dim / 256u;          /* q4_K 权重块数 */
    const uint64_t out_a_bytes = low_dim * kblocks * sizeof(cuda_block_q4_K);
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) return 0;
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a_q4k"));
    if (!out_a) return 0;
    /* 09-07: 分组 tile 核(cuda_q4k_tile.inc.cu)取代 16-lane dp4a 核, 激活仍是 q8_0 32 值块(精度不变); 容差级(结合律)。 */
    if (!q4k_tile_supported((uint32_t)kblocks, (uint32_t)low_dim)) {
        fprintf(stderr, "ds4: attn_output_a q4_K: 每行 %llu 块/低维 %llu 不在 tile 核支持的形状\n",
                (unsigned long long)kblocks, (unsigned long long)low_dim);
        return 0;
    }
    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    void *tmp = cuda_tmp_alloc(scale_offset + x_rows * blocks_a * sizeof(float), "attention output low q4k prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    ds4_launch_pdl(quantize_q8_0_f32_kernel, dim3((unsigned)blocks_a, (unsigned)x_rows, 1), 32, 0, 0,
                   xq, xscale, (const float *)heads->ptr, group_dim, blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q4k prequant launch")) return 0;
    return q4k_tile_grouped_launch((float *)low->ptr, (const char *)out_a, xq, xscale, (uint32_t)kblocks,
                                   (uint32_t)rank, n_groups, 1u);
}
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    ds4_launch_pdl(swiglu_kernel, (n + 255) / 256, 256, 0, g_cur_stream, (float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
