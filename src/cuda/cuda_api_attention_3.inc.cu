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
    /* q4_K 版 attn_output_a(与 q8 入口同构): 激活预量化仍是 q8_0 32 块, 权重行
     * = group_dim/256 个 q4_K 块, dot 走 dev_dot_q4_K_q8_0x8。 */
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
    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q4k prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)heads->ptr,
                                            group_dim, blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q4k prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 15u) / 16u, 1, 1);
    if (use_dp4a && kblocks <= 16u) {
        grouped_q4_K_a_preq_warp8_dp4a_kernel<<<grid_a, 256, (size_t)16u * (size_t)kblocks * 9u * sizeof(uint4)>>>((float *)low->ptr, out_a, xq, xscale,
                                                          group_dim, rank, n_groups, 1,
                                                          kblocks);
    } else {
        grouped_q4_K_a_preq_warp8_kernel<<<grid_a, 256, (kblocks <= 16u) ? (size_t)16u * (size_t)kblocks * 9u * sizeof(uint4) : 0>>>((float *)low->ptr, out_a, xq, xscale,
                                                          group_dim, rank, n_groups, 1,
                                                          kblocks, use_dp4a);
    }
    if (((const char *)0) /* DS4_AO_PROBE: 诊断开关已删(2026-08-22) */) {
        static int once = 0;
        if (once++ < 96) {
            (void)cudaDeviceSynchronize();
            float lo[4]; float *xh = (float *)malloc(group_dim * sizeof(float));
            (void)cudaMemcpy(lo, low->ptr, sizeof(lo), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(xh, heads->ptr, group_dim * sizeof(float), cudaMemcpyDeviceToHost);
            /* host 参考: row0 属 group0, 用原始 mmap 上的 q4_K 权重逐块 dequant×x */
            const uint8_t *w0 = (const uint8_t *)model_map + out_a_offset;   /* row0 */
            double ref = 0.0;
            for (uint64_t b = 0; b < kblocks; b++) {
                const uint8_t *blk = w0 + b * 144u;
                uint16_t hd, hm; memcpy(&hd, blk + 0, 2); memcpy(&hm, blk + 2, 2);
                const float d = dev_host_f16(hd), dmin = dev_host_f16(hm);
                const uint8_t *scales = blk + 4; const uint8_t *qs = blk + 16;
                for (uint32_t j = 0; j < 8u; j++) {
                    uint8_t sc, m;
                    if (j < 4u) { sc = scales[j] & 63u; m = scales[j + 4u] & 63u; }
                    else { sc = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
                           m = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
                    const uint32_t byte_off = (j >> 1u) * 32u;
                    const int shift = (int)(j & 1u) * 4;
                    for (uint32_t i = 0; i < 32u; i++) {
                        const int q4 = (qs[byte_off + i] >> shift) & 0xF;
                        ref += (double)(d * sc * q4 - dmin * m) * xh[b * 256u + j * 32u + i];
                    }
                }
            }
            /* 跨组检查: group1 首行(row=rank), 激活用 heads 第二段 */
            float lo_r1 = 0; (void)cudaMemcpy(&lo_r1, (const char *)low->ptr + rank * sizeof(float), 4, cudaMemcpyDeviceToHost);
            float *xh1 = (float *)malloc(group_dim * sizeof(float));
            (void)cudaMemcpy(xh1, (const char *)heads->ptr + group_dim * sizeof(float), group_dim * sizeof(float), cudaMemcpyDeviceToHost);
            const uint8_t *w1 = (const uint8_t *)model_map + out_a_offset + (uint64_t)rank * kblocks * 144u;
            double ref1 = 0.0;
            for (uint64_t b = 0; b < kblocks; b++) {
                const uint8_t *blk = w1 + b * 144u;
                uint16_t hd, hm; memcpy(&hd, blk + 0, 2); memcpy(&hm, blk + 2, 2);
                const float d = dev_host_f16(hd), dmin = dev_host_f16(hm);
                const uint8_t *scales = blk + 4; const uint8_t *qs = blk + 16;
                for (uint32_t j = 0; j < 8u; j++) {
                    uint8_t sc, m;
                    if (j < 4u) { sc = scales[j] & 63u; m = scales[j + 4u] & 63u; }
                    else { sc = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
                           m = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
                    const uint32_t byte_off = (j >> 1u) * 32u;
                    const int shift = (int)(j & 1u) * 4;
                    for (uint32_t i = 0; i < 32u; i++) {
                        const int q4 = (qs[byte_off + i] >> shift) & 0xF;
                        ref1 += (double)(d * sc * q4 - dmin * m) * xh1[b * 256u + j * 32u + i];
                    }
                }
            }
            fprintf(stderr, "ds4: [ao-probe] gpu_low0=%.4f host=%.4f | g1: gpu=%.4f host=%.4f\n",
                    lo[0], ref, lo_r1, ref1);
            fflush(stderr); free(xh); free(xh1);
        }
    }
    return cuda_ok(cudaGetLastError(), "attention_output_low_q4k launch");
}
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
