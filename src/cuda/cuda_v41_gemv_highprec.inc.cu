/* cuda_v41_gemv_highprec.inc.cu — ds4_cuda.cu 分片: **非 FP4 权重**的解码小批 GEMV(2026-09-15, clear.md C1/C2)。
 *
 * 谁在这儿: V4.1 里还没变成 FP4 的那几个矩阵 ——
 *   BF16(官方原生精度, C1 起不再展开成 f32): 路由 gate、compressor kv/gate、indexer wk/proj;
 *   F32(官方原件就是 f32): mHC 的 hc_attn_fn / hc_ffn_fn。
 * 它们每 token 只有 0.25 GB, 却曾经吃掉 3.25 ms —— 因为 n=1 时 cuBLAS 把一发 Sgemm 拆成
 * 47 发 gemvx + 88 发 dot + 88 发 reduce_1Block, 每 token 200 多次 launch。这里按骨架 GEMV
 * 同一形态自己写: 一 block 8 warp, 一 warp 一行(ksplit>1 时几个 warp 分 K 段), lane 走 128 位读。
 *
 * ★这个文件是过渡件★: clear.md C1 的 A/B 项要把 gate / hc 也压到 FP4, 过了五指标门之后
 * 这两个核就该被 v41_fp4x32_gemv 顶掉, 连同文件一起删。别在这儿加新功能。
 *
 * ★数值★: 与 cuBLAS 只差累加序(都是升 f32 相乘、f32 累加, 不碰 bf16 格点)。gate 的 top-6
 * 在近似并列时可能翻 ⇒ 门是主尺 NLL/PPL, 不是逐位。
 * ★in_dim 的整除要求★: f32 版 128 的倍数、bf16 版 256 的倍数(一个 warp 一轮走的元素数)。
 * 现有调用点全是 5120 / 512。不满足直接返回 0 让调用方硬失败, 不补标量尾巴 —— 补了就是给
 * 未来的新形状留一条悄悄变慢的路。 */

/* ---- f32 权重 GEMV(clear.md C2 ③: 把解码路上的 cublasSgemm 请出去) ----
 * 谁在用: 路由 gate(每层一发)、compressor kv/gate(4 个 kv 源层)、indexer wk/proj —— §2 精度表
 * 里还留在 f32 的那几个矩阵。它们加起来每 token 只有 0.40 GB, 但 09-15 nsys 实测吃 3.25 ms:
 * n=1 时 cuBLAS 把一发 Sgemm 拆成 47 发 gemvx + 88 发 dot_kernel + 88 发 reduce_1Block,
 * 每 token 200 多次 launch, 单核带宽只有 123 GB/s(墙的 51%)。
 * 这里按骨架 GEMV 同一形态写: 一 block 8 warp, 一 warp 一行(ksplit>1 时几个 warp 分 K 段),
 * lane 走 float4 —— 每 4 个元素 1 次 128 位权重读 + 1 次 128 位激活读 + 4 条 FFMA。
 * ★数值★: 与 Sgemm 只差累加序(都是 f32 乘 f32 累加, 不碰 bf16 格点)。gate 的 top-6 选择在
 * 近似并列时可能翻 ⇒ 门是主尺 NLL/PPL, 不是逐位。
 * ★in_dim 必须是 128 的倍数★: 现有调用点全是 5120 / 512。不满足直接返回 0 让调用方硬失败,
 * 不补标量尾巴 —— 补了就是给未来的新形状留一条悄悄变慢的路。 */
template <uint32_t NT>
__global__ static void v41_f32_gemv_kernel(float *out, const float *w, const float *x, uint32_t in_dim,
                                           uint32_t out_dim, uint32_t x_stride, uint32_t out_stride, uint32_t ksplit) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const float *wr = w + (uint64_t)r * in_dim;
        for (uint32_t c = kpart * 128u + lane * 4u; c < in_dim; c += ksplit * 128u) {
            const float4 wv = *(const float4 *)(wr + c);
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                const float4 xv = *(const float4 *)(x + (uint64_t)t * x_stride + c);
                acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_stride + r] = acc[t];
    }
}
/* ---- bf16 权重 GEMV(clear.md C1: 路由 gate / compressor / indexer 投影存回 BF16 之后) ----
 * 与上面的 f32 版同一形态, 只是权重每个元素 2 字节 ⇒ 一次 128 位读拿 8 个(f32 版拿 4 个),
 * 所以一个 warp 一轮走 256 个元素。要求 in_dim 是 256 的倍数 —— 现有调用点是 5120 / 512。
 * ★数值★: bf16 权重升 f32 再乘 f32 激活, f32 累加。与"f32 存的同一份权重"逐值相同
 * (存 BF16 是把原件原样搬过来, 不是压缩), 只差累加序 ⇒ 门是主尺 NLL/PPL。 */
template <uint32_t NT>
__global__ static void v41_bf16_gemv_kernel(float *out, const __nv_bfloat16 *w, const float *x, uint32_t in_dim,
                                            uint32_t out_dim, uint32_t x_stride, uint32_t out_stride, uint32_t ksplit) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const __nv_bfloat16 *wr = w + (uint64_t)r * in_dim;
        for (uint32_t c = kpart * 256u + lane * 8u; c < in_dim; c += ksplit * 256u) {
            const uint4 raw = *(const uint4 *)(wr + c);
            __nv_bfloat162 wb[4];
            memcpy(wb, &raw, 16);
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                const float4 x0 = *(const float4 *)(x + (uint64_t)t * x_stride + c);
                const float4 x1 = *(const float4 *)(x + (uint64_t)t * x_stride + c + 4u);
                const float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
                const float2 w2 = __bfloat1622float2(wb[2]), w3 = __bfloat1622float2(wb[3]);
                acc[t] += w0.x * x0.x + w0.y * x0.y + w1.x * x0.z + w1.y * x0.w
                        + w2.x * x1.x + w2.y * x1.y + w3.x * x1.z + w3.y * x1.w;
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_stride + r] = acc[t];
    }
}
static int v41_bf16_gemv(const __nv_bfloat16 *w, uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                         uint32_t n_tok, const char *what) {
    if ((in_dim % 256u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)(in_dim / 256u);
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), 1);
    #define V41_BFGEMV_LAUNCH(NT) v41_bf16_gemv_kernel<NT><<<grid, 256, 0, g_cur_stream>>>( \
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)out_dim, ksplit)
    switch (n_tok) {
        case 1: V41_BFGEMV_LAUNCH(1u); break;  case 2: V41_BFGEMV_LAUNCH(2u); break;
        case 3: V41_BFGEMV_LAUNCH(3u); break;  case 4: V41_BFGEMV_LAUNCH(4u); break;
        case 5: V41_BFGEMV_LAUNCH(5u); break;  case 6: V41_BFGEMV_LAUNCH(6u); break;
        case 7: V41_BFGEMV_LAUNCH(7u); break;  default: V41_BFGEMV_LAUNCH(8u); break;
    }
    #undef V41_BFGEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

static int v41_f32_gemv(const float *w, uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                        uint32_t n_tok, const char *what) {
    if ((in_dim % 128u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)(in_dim / 128u);   /* K 的分段单位 = 128 个元素 */
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), 1);
    #define V41_F32GEMV_LAUNCH(NT) v41_f32_gemv_kernel<NT><<<grid, 256, 0, g_cur_stream>>>( \
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)out_dim, ksplit)
    switch (n_tok) {
        case 1: V41_F32GEMV_LAUNCH(1u); break;  case 2: V41_F32GEMV_LAUNCH(2u); break;
        case 3: V41_F32GEMV_LAUNCH(3u); break;  case 4: V41_F32GEMV_LAUNCH(4u); break;
        case 5: V41_F32GEMV_LAUNCH(5u); break;  case 6: V41_F32GEMV_LAUNCH(6u); break;
        case 7: V41_F32GEMV_LAUNCH(7u); break;  default: V41_F32GEMV_LAUNCH(8u); break;
    }
    #undef V41_F32GEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}


/* ---- FP8(e4m3 + 32×32 块 ue8m0)权重 GEMV: engram wkv(clear.md C1) ----
 * 这张表 6144×25600 两层共 0.315 GB, 是解码每 token 第四大的读量。转换器原来把它展开成 f16
 * 落盘(0.63 GB, 而且 f16 的 10 位尾数装不下 e4m3×2^k 的全部取值), 现在盘上就是官方的 e4m3 平面 +
 * 缩放平面 —— 字节减半, 值反而更准。
 * 布局: w[0 .. rows*cols) 是 e4m3; 紧跟 ceil(rows/32)*ceil(cols/32) 个 ue8m0, 行块 r/32、列块 c/32。
 * 一个 lane 一轮吃 16 个元素(一次 128 位读), 一个 warp 一轮走 512 —— 512 是 32 的倍数, 所以
 * lane 的 16 个元素必定落在同一个列块里, 缩放只查一次。要求 in_dim 是 512 的倍数(wkv 是 6144)。 */
template <uint32_t NT>
__global__ static void v41_fp8blk_gemv_kernel(float *out, const uint8_t *w, const uint8_t *sc, const float *x,
                                              uint32_t in_dim, uint32_t out_dim, uint32_t sbc, uint32_t ksplit) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const uint8_t *wr = w + (uint64_t)r * in_dim;
        const uint8_t *scr = sc + (uint64_t)(r >> 5) * sbc;
        for (uint32_t c = kpart * 512u + lane * 16u; c < in_dim; c += ksplit * 512u) {
            const uint4 raw = *(const uint4 *)(wr + c);
            uint8_t b[16];
            memcpy(b, &raw, 16);
            const float s = ds4_e8m0_to_f32(scr[c >> 5]);
            float wv[16];
            #pragma unroll
            for (uint32_t j = 0; j < 16u; j++) wv[j] = ds4_e4m3fn_to_f32(b[j]) * s;
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                const float *xt = x + (uint64_t)t * in_dim + c;
                #pragma unroll
                for (uint32_t j = 0; j < 16u; j++) acc[t] += wv[j] * xt[j];
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_dim + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_dim + r] = acc[t];
    }
}
/* 预填(n > 8): 先把这一张表解成 bf16 再一发 cuBLAS。
 * ★为什么这里允许落暂存, 别处不允许★: e4m3 带 32×32 二维块缩放, cuBLASLt 的 FP8 只吃"整张一个缩放",
 * 没有算法能直接喂; 而这张表只有 0.157 GB/层、全模型只有两层, 解一次摊到整块 token 上。
 * 解码路(上面的 GEMV)一个字节都不落暂存。 */
__global__ static void v41_fp8blk_to_bf16_kernel(__nv_bfloat16 *o, const uint8_t *w, const uint8_t *sc,
                                                 uint32_t in_dim, uint32_t sbc, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t r = (uint32_t)(i / in_dim), c = (uint32_t)(i % in_dim);
    o[i] = __float2bfloat16(ds4_e4m3fn_to_f32(w[i]) * ds4_e8m0_to_f32(sc[(uint64_t)(r >> 5) * sbc + (c >> 5)]));
}

static int v41_fp8blk_gemv(const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t out_dim,
                           const float *x, float *out, uint32_t n_tok, const char *what) {
    if ((in_dim % 512u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)(in_dim / 512u);
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit, sbc = (uint32_t)((in_dim + 31u) / 32u);
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), 1);
    #define V41_FP8GEMV_LAUNCH(NT) v41_fp8blk_gemv_kernel<NT><<<grid, 256, 0, g_cur_stream>>>( \
        out, w, sc, x, (uint32_t)in_dim, (uint32_t)out_dim, sbc, ksplit)
    switch (n_tok) {
        case 1: V41_FP8GEMV_LAUNCH(1u); break;  case 2: V41_FP8GEMV_LAUNCH(2u); break;
        case 3: V41_FP8GEMV_LAUNCH(3u); break;  case 4: V41_FP8GEMV_LAUNCH(4u); break;
        case 5: V41_FP8GEMV_LAUNCH(5u); break;  case 6: V41_FP8GEMV_LAUNCH(6u); break;
        case 7: V41_FP8GEMV_LAUNCH(7u); break;  default: V41_FP8GEMV_LAUNCH(8u); break;
    }
    #undef V41_FP8GEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}
static int v41_fp8blk_to_bf16(__nv_bfloat16 *o, const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t rows) {
    const uint64_t n = in_dim * rows;
    const uint32_t sbc = (uint32_t)((in_dim + 31u) / 32u);
    v41_fp8blk_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>(o, w, sc, (uint32_t)in_dim, sbc, n);
    return cuda_ok(cudaGetLastError(), "v41 fp8blk→bf16");
}
