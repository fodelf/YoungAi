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
    ds4_launch_pdl(quantize_q8_0_f32_kernel, qgrid, 32, 0, 0, xq, xscale, (const float *)x->ptr, in_dim, blocks);
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
    ds4_launch_pdl(quantize_q8_0_f32_kernel, (unsigned)blocks, 32, 0, 0, xq, xscale, (const float *)x->ptr, in_dim, blocks);
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
    /* 09-07: 整行 stage 整块/lane 核(cuda_q4k_tile.inc.cu), 激活仍是 q8_0 32 值块(精度不变)。 */
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    void *tmp = cuda_tmp_alloc(scale_offset + blocks * sizeof(float), "q4k hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    ds4_launch_pdl(quantize_q8_0_f32_kernel, (unsigned)blocks, 32, 0, 0, xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q4k_hc_expand quantize launch")) return 0;
    return q4k_hc_expand_launch((float *)out_hc->ptr, (float *)block_out->ptr, (const float *)residual_hc->ptr,
                                (const float *)split->ptr, wptr, xq, xscale, (uint32_t)kblocks, (uint32_t)out_dim, n_embd, n_hc);
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
        ds4_launch_pdl(f32_to_f16_kernel, (xh_count + 255) / 256, 256, 0, 0, xh, (const float *)x->ptr, xh_count);
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
    /* split-K 缓冲预分配(必须在非 capture 调用里做, 见 down partial 同款注释) */
    if (!g_f16sk_partial) {
        cudaStreamCaptureStatus fcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &fcs);
        if (fcs == cudaStreamCaptureStatusNone) cuda_decode_scratch_prepare();
    }
    if (n_tok <= 8 && !serial_f16 && !serial_router &&
        (n_tok == 1 || 1) &&
        out_dim <= 512u && in_dim >= 4096u && g_f16sk_partial) {
        /* out≤64: 多段 split-K + 固定序 reduce; 64<out≤512: S=1 单段=一行一块直写
         * (原 8 行/块只发 out/8 个块, out=256 时 32 块=1/6 GPU)。 */
        /* 09-06 K4 消融(负, 已回退): out≤64 的窄输出(hc_fn 16384→24)把段数 8→64 想靠更多块补
         * 占用, 实测每发 8.3→10.0 µs、reduce 1.3→1.9 µs(+0.28 ms/token): 这类 0.8 MB 权重
         * 本来就在 L2 里, 多段只是多了部分和写读。别再往段数上加。 */
        uint32_t S = (uint32_t)((192u + out_dim - 1u) / out_dim);
        if (S > 64u) S = 64u;
        if ((uint64_t)S * out_dim > 4096u) S = (uint32_t)(4096u / out_dim);
        if (S == 0) S = 1;
        uint32_t chunk = (uint32_t)(((in_dim + (uint64_t)S * 8u - 1u) / ((uint64_t)S * 8u)) * 8u);
        float *pdst = (S == 1) ? (float *)out->ptr : g_f16sk_partial;
        unsigned skcells = (unsigned)out_dim * S * (unsigned)n_tok;
        if (skcells > 1024u) skcells = 1024u;
        ds4_launch_pdl(matmul_f16_splitk_kernel, skcells, 256, 0, g_cur_stream, pdst, w, (const float *)x->ptr,
                                                  in_dim, out_dim, chunk, S, (uint32_t)n_tok);
        if (S > 1) {
            const unsigned rn = (unsigned)(out_dim * n_tok);
            ds4_launch_pdl(matmul_f16_splitk_reduce_kernel, (rn + 255u) / 256u, 256, 0, g_cur_stream, 
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

/* decode scratch 预建(非 capture 时机调用): split-K f16 partial [n_tok<=8][S*out<=4096]。
 * 09-05 定罪: token graph 开着时第一次解码就在 capture 里, 这里的惰性 cudaMalloc 被闸掉 ⇒
 * hc_fn(16384→24) 走非 split-K 归约序, 与直发差几个 ulp, 43 层后路由翻转 ⇒ 图/直发分叉。 */
static void cuda_decode_scratch_prepare(void) {
    cuda_attn_split_scratch_prepare();   /* 解码注意力分行核部分和(cuda_api_attention_4) */
    rtk_scratch_prepare();               /* indexer top-k 多 block 版暂存(cuda_indexer_kernels_6) */
    if (g_f16sk_partial) return;
    if (cudaMalloc(&g_f16sk_partial, 8u * 4096u * sizeof(float)) != cudaSuccess) {
        g_f16sk_partial = NULL; (void)cudaGetLastError();
    }
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
    /* 2 行/块版判负存档(原 DS4_F16_PAIR2 旋钮, 定死关): 实测 2778→2866 反向(多行族二败); 核末参 0 = 1 行/块, 网格 2 × out_dim */
    const uint32_t fp2 = 0u;
    unsigned pgrid = (unsigned)(2u * out_dim);
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

