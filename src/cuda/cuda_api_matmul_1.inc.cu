/* cuda_api_matmul_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * matmul q8_0/f16/f32 与 norm/rope GPU API。
 */
int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    if ((in_dim & 31u) == 0u) {
        const cuda_q8r_entry *r0 = cuda_q8r_get(model_map, weight0_offset, out0_dim, blocks);
        const cuda_q8r_entry *r1 = r0 ? cuda_q8r_get(model_map, weight1_offset, out1_dim, blocks) : NULL;
        if (r0 && r1) {
            matmul_q8r_pair_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
                    (float *)out0->ptr, (float *)out1->ptr,
                    r0->scales, r0->qs, r1->scales, r1->qs,
                    xq, xscale, out0_dim, out1_dim, blocks);
            return cuda_ok(cudaGetLastError(), "matmul_q8r pair launch");
        }
    }
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) return 0;
    if ((in_dim & 31u) == 0u) {
        const cuda_q8r_entry *re = cuda_q8r_get(model_map, weight_offset, out_dim, blocks);
        if (re) {
            matmul_q8r_hc_expand_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                    (float *)out_hc->ptr, (float *)block_out->ptr,
                    block_add ? (const float *)block_add->ptr : NULL,
                    (const float *)residual_hc->ptr, (const float *)split->ptr,
                    re->scales, re->qs, xq, xscale,
                    out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "matmul_q8r hc launch");
        }
    }
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

int ds4_gpu_matmul_q4_K_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* q4_K 版 attn_output_b + hc expand(与 q8 labeled 版同构, epilogue 逐字同义)。 */
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd || in_dim % 256u != 0) {
        return 0;
    }
    const uint64_t blocks = in_dim / 32u;
    const uint64_t kblocks = in_dim / 256u;
    const uint64_t weight_bytes = out_dim * kblocks * sizeof(cuda_block_q4_K);
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q4k_hc_expand");
    if (!wptr) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q4k hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q4k_hc_expand quantize launch")) return 0;
    if (use_dp4a && kblocks <= 32u) {
        matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, (size_t)8u * (size_t)((kblocks > 16u) ? 16u : kblocks) * 9u * sizeof(uint4)>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                (const float *)block_out->ptr,   /* has_add=0, 占位 */
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                kblocks,
                0);
    } else {
        matmul_q4_K_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, (kblocks <= 32u) ? (size_t)8u * (size_t)((kblocks > 16u) ? 16u : kblocks) * 9u * sizeof(uint4) : 0>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                (const float *)block_out->ptr,   /* has_add=0, 占位 */
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                kblocks,
                0,
                use_dp4a);
    }
    if (((const char *)0) /* DS4_AO_PROBE: 诊断开关已删(2026-08-22) */) {
        static int onceb = 0;
        if (onceb++ < 96) {
            (void)cudaDeviceSynchronize();
            float bo0 = 0; float *xh = (float *)malloc(in_dim * sizeof(float));
            (void)cudaMemcpy(&bo0, block_out->ptr, 4, cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(xh, x->ptr, in_dim * sizeof(float), cudaMemcpyDeviceToHost);
            const uint8_t *w0 = (const uint8_t *)model_map + weight_offset;
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
            /* out_hc[0][0] 校验: block_v*post[0] + Σ comb[0+src*n_hc]*res[src][0] */
            float oh0 = 0; (void)cudaMemcpy(&oh0, out_hc->ptr, 4, cudaMemcpyDeviceToHost);
            float spl[80]; float res0[4];
            (void)cudaMemcpy(spl, split->ptr, sizeof(float) * (2u * n_hc + n_hc * n_hc), cudaMemcpyDeviceToHost);
            for (uint32_t s = 0; s < n_hc && s < 4u; s++)
                (void)cudaMemcpy(&res0[s], (const char *)residual_hc->ptr + (uint64_t)s * n_embd * 4, 4, cudaMemcpyDeviceToHost);
            double ohref = ref * spl[n_hc];
            for (uint32_t s = 0; s < n_hc; s++) ohref += spl[2u * n_hc + 0u + s * n_hc] * res0[s];
            int hcnan = 0; float hcmax = 0;
            {   const uint64_t hn = (uint64_t)n_hc * n_embd;
                float *hf = (float *)malloc(hn * 4);
                if (hf && cudaMemcpy(hf, out_hc->ptr, hn * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
                    for (uint64_t i2 = 0; i2 < hn; i2++) {
                        if (hf[i2] != hf[i2]) hcnan++;
                        else if (fabsf(hf[i2]) > hcmax) hcmax = fabsf(hf[i2]);
                    }
                free(hf);
            }
            fprintf(stderr, "ds4: [ao-b#%02d] bo0 g=%.3f h=%.3f | hc g=%.3f h=%.3f | hc_nan=%d hc_max=%.1f\n",
                    onceb - 1, bo0, ref, oh0, ohref, hcnan, hcmax);
            fflush(stderr); free(xh);
        }
    }
    return cuda_ok(cudaGetLastError(), "matmul_q4k_hc_expand launch");
}

int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = 0;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        0;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        (n_tok == 1u || (n_tok <= 8u && 1)) &&
        1;
    /* 小批数值对齐(2026-08-21): verify 批(k<=8)改走 decode 的有序 kernel —— cublas 会把
     * 激活降成 f16(10 位尾数)再算, 与单 token 的 fp32 有序路差 ~1e-5, 逐层放大后 10% 位置
     * argmax 翻转, 直接吃掉投机接受率。prefill(大批)仍走 cublas: 那里吞吐优先。 */
    const int f16_exact_batch = (n_tok > 1 && n_tok <= 8 &&
                                 1);
    if (!serial_f16 && g_cublas_ready && n_tok > 1 && !f16_exact_batch) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    /* 诊断探针(DS4_F16_DIMS=1): 打印每个唯一 (in,out,路径), 用后即弃 */
    if (((const char *)0) /* DS4_F16_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
        static uint64_t seen[64][2]; static int nseen = 0;
        int hit = 0;
        for (int i = 0; i < nseen; i++) if (seen[i][0] == in_dim && seen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && nseen < 64) {
            seen[nseen][0] = in_dim; seen[nseen][1] = out_dim; nseen++;
            fprintf(stderr, "ds4: [f16-dims] in=%llu out=%llu serial=%d ord=%d ntok=%llu\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim,
                    (int)(serial_f16 || serial_router), (int)ordered_router, (unsigned long long)n_tok);
        }
    }
    /* split-K 缓冲预分配(必须在非 capture 调用里做, 见 down partial 同款注释) */
    if (!g_f16sk_partial) {
        cudaStreamCaptureStatus fcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &fcs);
        if (fcs == cudaStreamCaptureStatusNone) {
            /* [n_tok<=8][S*out<=4096] */
            if (cudaMalloc(&g_f16sk_partial, 8u * 4096u * sizeof(float)) != cudaSuccess) {
                g_f16sk_partial = NULL; (void)cudaGetLastError();
            }
        }
    }
    if (n_tok <= 8 && !serial_f16 && !serial_router &&
        (n_tok == 1 || 1) &&
        out_dim <= 512u && in_dim >= 4096u && g_f16sk_partial) {
        /* out≤64: 多段 split-K + 固定序 reduce; 64<out≤512: S=1 单段=一行一块直写
         * (原 8 行/块只发 out/8 个块, out=256 时 32 块=1/6 GPU)。 */
        uint32_t S = (uint32_t)((192u + out_dim - 1u) / out_dim);
        if (S > 64u) S = 64u;
        if ((uint64_t)S * out_dim > 4096u) S = (uint32_t)(4096u / out_dim);
        if (S == 0) S = 1;
        uint32_t chunk = (uint32_t)(((in_dim + (uint64_t)S * 8u - 1u) / ((uint64_t)S * 8u)) * 8u);
        float *pdst = (S == 1) ? (float *)out->ptr : g_f16sk_partial;
        unsigned skcells = (unsigned)out_dim * S * (unsigned)n_tok;
        if (skcells > 1024u) skcells = 1024u;
        matmul_f16_splitk_kernel<<<skcells, 256, 0, g_cur_stream>>>(pdst, w, (const float *)x->ptr,
                                                  in_dim, out_dim, chunk, S, (uint32_t)n_tok);
        if (S > 1) {
            const unsigned rn = (unsigned)(out_dim * n_tok);
            matmul_f16_splitk_reduce_kernel<<<(rn + 255u) / 256u, 256, 0, g_cur_stream>>>(
                    (float *)out->ptr, g_f16sk_partial, (uint32_t)out_dim, S, (uint32_t)n_tok);
        }
        return cuda_ok(cudaGetLastError(), "matmul_f16_splitk launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (ordered_router) {
        dim3 ogrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_f16_ordered_chunks_kernel<<<ogrid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

int ds4_gpu_matmul_f16_pair_tensor(
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
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1 ||
        0 ||
        0 ||
        0 ||
        0) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (((const char *)0) /* DS4_F16_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {   /* 诊断探针: pair 形状(与单矩阵入口同款) */
        static uint64_t pseen[64][2]; static int pn = 0;
        int hit = 0;
        for (int i = 0; i < pn; i++) if (pseen[i][0] == in_dim && pseen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && pn < 64) {
            pseen[pn][0] = in_dim; pseen[pn][1] = out_dim; pn++;
            fprintf(stderr, "ds4: [f16-pair-dims] in=%llu out=%llu (grid=%u blk=256)\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, (unsigned)(2u * out_dim));
        }
    }
    static uint32_t fp2 = 99u;
    if (fp2 == 99u) { const char *e = ((const char *)0) /* DS4_F16_PAIR2: 路径开关已删(2026-08-22 隐形炸弹清理) */; fp2 = e ? (uint32_t)atoi(e) : 0u; }   /* 默认关: 2行版实测2778→2866反向(多行族二败) */
    unsigned pgrid = fp2 ? (unsigned)out_dim : (unsigned)(2u * out_dim);
    if (pgrid > ds4_grid_cap()) pgrid = ds4_grid_cap();   /* pair3(施工日2): 常驻块行循环, 灭块级碎片 */
    matmul_f16_pair_rowblock_kernel<<<pgrid,
                                      256, 0, g_cur_stream>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        fp2);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_rowblock launch");
}

