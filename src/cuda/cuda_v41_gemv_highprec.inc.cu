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
/* XB=1: 激活从 bf16 缓冲读 —— 与 cuda_v41_4.inc.cu 的骨架 GEMV 同一刀、同一笔账(见那边核头的注释)。
 * 这里一个 lane 一轮要 16 个元素: f32 是 64 B ⇒ 32 个 lane 铺开 2048 B = 16 个波前/token;
 * bf16 是 32 B ⇒ 1024 B = 8 个。NT=6 时省 48 个波前, 而权重侧一轮才 4 个。 */
/* ★grid.y = 块对角的"第几组"(2026-09-17)★: DSpark 三塔的 wo_a 是 8 组块对角, 而它在原件里是
 * FP8 —— 组 g 的行段 [g·out_dim, (g+1)·out_dim) 在权重平面里是连续的, 缩放平面同样连续
 * (out_dim 是 32 的倍数, 所以行块也整除)。所以不用另写核, 加一个行偏移就够。
 * 非分组调用传 n_groups=1, 一切照旧(偏移恒 0), 与改之前逐条指令相同。 */
template <uint32_t NT, uint32_t XB>
__global__ static void v41_fp8blk_gemv_kernel(float *out, const uint8_t *w, const uint8_t *sc, const float *x,
                                              const __nv_bfloat16 *x16,
                                              uint32_t in_dim, uint32_t out_dim, uint32_t sbc, uint32_t ksplit,
                                              uint32_t x_stride, uint32_t out_stride,
                                              uint32_t x_gstride, uint32_t out_gstride) {
    const uint32_t g = blockIdx.y;
    w += (uint64_t)g * in_dim * out_dim;
    sc += (uint64_t)g * ((out_dim + 31u) / 32u) * sbc;
    x += (uint64_t)g * x_gstride; out += (uint64_t)g * out_gstride;
    if (XB) x16 += (uint64_t)g * x_gstride;
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
                /* ★两条路各写各的★: 共用一个 float xv[16] 中间数组会改整只核的寄存器分配
                 * (骨架 GEMV 那边同样的写法实测让纯解码慢 5%), XB=0 这一支逐字抄回改造前的样子。 */
                if (XB) {   /* 16 个 bf16 = 32 B 连续, 两条 uint4; 补零还原成 f32 是恒等的 */
                    const uint4 *xp = (const uint4 *)(x16 + (uint64_t)t * x_stride + c);
                    const uint4 x0 = xp[0], x1 = xp[1];
                    const uint32_t xu[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; j++) {
                        acc[t] += wv[2u * j]      * __uint_as_float(xu[j] << 16);
                        acc[t] += wv[2u * j + 1u] * __uint_as_float(xu[j] & 0xffff0000u);
                    }
                } else {
                    const float *xt = x + (uint64_t)t * x_stride + c;
                    #pragma unroll
                    for (uint32_t j = 0; j < 16u; j++) acc[t] += wv[j] * xt[j];
                }
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

/* ★in_dim 只要 32 的倍数就行(2026-09-17 放宽, 原来要 512)★: 一个 lane 一轮吃 16 个元素、
 * 一个 warp 一轮走 512, 尾轮由 `c < in_dim` 自然收口 —— 512 不整除只是最后一轮有 lane 闲着,
 * **不影响正确性**(缩放按 32 一块查, 32 整除就对齐)。放宽是为了三塔的 wq_b(in_dim 1280)。
 * engram 那条(in_dim 6144)走的分支一个字没变。 */
static int v41_fp8blk_gemv_g(const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t out_dim,
                             const float *x, float *out, uint32_t n_tok, uint32_t n_groups,
                             uint32_t x_stride, uint32_t out_stride, uint32_t x_gstride, uint32_t out_gstride,
                             const char *what) {
    if ((in_dim % 32u) != 0u || (in_dim % 16u) != 0u || n_tok == 0 || n_groups == 0) return 0;
    if (n_tok > V41_GEMV_MAX_TOK) {
        /* ★批大于模板上限就按 8 行一段循环★(2026-09-18 实撞): 草稿器整窗重建要把 128 个位置一次过 main_proj/wkv,
         * 以前这里直接 return 0 且没人出声 —— 预填后第一轮草稿从来没成功过, 三塔窗口里从来没有提示的上下文。
         * 权重多读几遍(128 行 = 16 遍)只发生在整窗重建那一次, 逐行推进仍是单发。 */
        for (uint32_t off = 0; off < n_tok; off += V41_GEMV_MAX_TOK) {
            const uint32_t nn = n_tok - off < V41_GEMV_MAX_TOK ? n_tok - off : V41_GEMV_MAX_TOK;
            if (!v41_fp8blk_gemv_g(w, sc, in_dim, out_dim, x + (uint64_t)off * x_stride, out + (uint64_t)off * out_stride,
                                   nn, n_groups, x_stride, out_stride, x_gstride, out_gstride, what)) return 0;
        }
        return 1;
    }
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit * n_groups < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)((in_dim + 511u) / 512u);
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit, sbc = (uint32_t)((in_dim + 31u) / 32u);
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), n_groups);
    /* 验证批(NT>1)的激活转 bf16, 与骨架 GEMV 同一刀; 解码(NT=1)不转(那条是字节受限的) */
    const __nv_bfloat16 *x16 = NULL;
    if (n_tok > 1u) {
        const uint64_t xn = (uint64_t)n_tok * x_stride;
        __nv_bfloat16 *xbuf = (__nv_bfloat16 *)v41_grow(&g_v41_gemv_xb, xn * sizeof(__nv_bfloat16), "v41 fp8blk x bf16");
        if (!xbuf) return 0;
        v41_x_to_bf16_kernel<<<(unsigned)((xn + 255) / 256), 256, 0, g_cur_stream>>>(xbuf, x, xn);
        if (!cuda_ok(cudaGetLastError(), "v41 fp8blk x bf16")) return 0;
        x16 = xbuf;
    }
    #define V41_FP8GEMV_LAUNCH(NT, XB) v41_fp8blk_gemv_kernel<NT, XB><<<grid, 256, 0, g_cur_stream>>>( \
        out, w, sc, x, x16, (uint32_t)in_dim, (uint32_t)out_dim, sbc, ksplit, \
        x_stride, out_stride, x_gstride, out_gstride)
    switch (n_tok) {
        case 1: V41_FP8GEMV_LAUNCH(1u, 0u); break;  case 2: V41_FP8GEMV_LAUNCH(2u, 1u); break;
        case 3: V41_FP8GEMV_LAUNCH(3u, 1u); break;  case 4: V41_FP8GEMV_LAUNCH(4u, 1u); break;
        case 5: V41_FP8GEMV_LAUNCH(5u, 1u); break;  case 6: V41_FP8GEMV_LAUNCH(6u, 1u); break;
        case 7: V41_FP8GEMV_LAUNCH(7u, 1u); break;  default: V41_FP8GEMV_LAUNCH(8u, 1u); break;
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

/* ---- 出口头的逐列平方和(mtp-1.md M6′: 草稿器对齐改用**出口度量**) ----
 *
 * 为什么要它: 草稿器与主模型的 logits 由**同一个出口头** W 算(官方 forward_head 借主模型的 head),
 * 所以"把草稿器的出口隐态掰到主模型的那个"本该按 ‖W·(h_d − h_m)‖ 来量, 而不是按 ‖h_d − h_m‖。
 * 09-16 那次 M6 判负(留出一致率 0.4570 → 挂上反而 0.4336)的真因就在这: 纯 L2 把力气花在**方差大**的
 * 方向上, 而 argmax 是由 head 那 12.9 万行里的细微差额定的 —— 两者根本不是一个度量。
 *
 * 这里出的是 G = WᵀW 的**对角线**(每一列的平方和), 即"这个隐态维度被出口头放大多少"。
 * ★为什么只要对角线★: 整份 G 是 5120×5120(105 MB), 解算侧还要 Cholesky, 是另一档工程;
 * 对角线是它最省的一阶近似 —— 把"头基本不看的维度"压下去, 这一条就够解释那次判负了。
 * 真要上整份 G, 照这里的形态改成 cuBLAS syrk 即可(权重已经在设备上, 逐块解成 bf16)。
 *
 * W 在盘上是 fp4x32 [V][D] 行主序 ⇒ 列 d 的平方和 = Σ_v W[v][d]²。一 block 管一列,
 * 256 个线程沿 v 方向跨步累加再树形归约。f32 累加(V=129280 项, 值域 ≤ 6×2^e, f32 够用)。 */
__global__ static void v41_head_colnorm_kernel(float *out, const uint8_t *w, uint32_t V, uint32_t D) {
    const uint32_t d = blockIdx.x;
    const uint32_t nblk_row = D / 32u, blk = d >> 5, el = d & 31u;
    __shared__ float red[256];
    float s = 0.f;
    for (uint32_t v = threadIdx.x; v < V; v += blockDim.x) {
        const uint8_t *p = w + ((uint64_t)v * nblk_row + blk) * 17u;
        const float sc = ds4_e8m0_to_f32(p[16]);
        const uint8_t by = p[el >> 1];
        const float wv = ds4_fp4_nibble_to_f32((el & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu)) * sc;
        s += wv * wv;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (uint32_t k = blockDim.x >> 1; k; k >>= 1) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[d] = red[0];
}
int ds4_gpu_v41_head_colnorm_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                    uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd) {
    if (!out || (n_embd % 32u) || out->bytes < (uint64_t)n_embd * 4) return 0;
    const uint64_t nblk = (uint64_t)n_vocab * (n_embd / 32u);
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, nblk * 17u, "v41 head colnorm");
    if (!w) return 0;
    v41_head_colnorm_kernel<<<n_embd, 256, 0, g_cur_stream>>>((float *)out->ptr, w, n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "v41 head colnorm");
}

/* 单组入口(engram wkv 等): 与改造前同义, 组数 1、步长 = 维度。 */
static int v41_fp8blk_gemv(const uint8_t *w, const uint8_t *sc, uint64_t in_dim, uint64_t out_dim,
                           const float *x, float *out, uint32_t n_tok, const char *what) {
    return v41_fp8blk_gemv_g(w, sc, in_dim, out_dim, x, out, n_tok, 1u,
                             (uint32_t)in_dim, (uint32_t)out_dim, 0u, 0u, what);
}

/* ---- 三塔投影的 FP8 入口(mtp-1.md M6 真因修复, 2026-09-17) ----
 * 为什么要这两个: DSpark 三塔在原件里是 FP8(E4M3 + 32×32), 我们的转换器原来跟着主干一起压成 FP4,
 * 把草稿器的精度一并砍了(草稿器对 FP 原模型的首位一致率只剩 0.50)。现在盘上存回 FP8, 引擎按张量
 * 类型分发到这里。out 出口补一次 bf16 舍入 —— fp4 那条路是核里 round_out 折进去的, 两条要同口径。 */
int ds4_gpu_v41_matmul_fp8blk_round_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                           uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                           const ds4_gpu_tensor *x, uint32_t n_tok, int round_out) {
    if (!out || !x || n_tok == 0) return 0;
    const uint64_t sbc = (in_dim + 31u) / 32u, sbr = (out_dim + 31u) / 32u;
    const uint64_t wbytes = in_dim * out_dim + sbr * sbc;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *W = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 mtp fp8 w");
    if (!W) return 0;
    if (!v41_fp8blk_gemv_g(W, W + in_dim * out_dim, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr,
                           n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim, 0u, 0u, "v41 mtp fp8 gemv")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(out, (uint64_t)n_tok * out_dim) : 1;
}
int ds4_gpu_v41_grouped_matmul_fp8blk_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                             uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out) {
    if (!low || !heads || n_tok == 0 || n_groups == 0) return 0;
    const uint64_t in_all = (uint64_t)n_groups * group_dim, out_all = (uint64_t)n_groups * rank;
    const uint64_t sbc = (group_dim + 31u) / 32u, sbr = (rank + 31u) / 32u;
    /* 盘上是整块 [n_groups·rank][group_dim] 的 e4m3 平面 + 同形状的缩放平面 —— 组 g 的行段连续,
     * 所以核里按 grid.y 加个行偏移就够(见 v41_fp8blk_gemv_kernel 的注释)。 */
    const uint64_t wbytes = out_all * group_dim + n_groups * sbr * sbc;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *W = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 mtp wo_a fp8");
    if (!W) return 0;
    if (!v41_fp8blk_gemv_g(W, W + out_all * group_dim, group_dim, rank, (const float *)heads->ptr,
                           (float *)low->ptr, n_tok, n_groups, (uint32_t)in_all, (uint32_t)out_all,
                           (uint32_t)group_dim, (uint32_t)rank, "v41 mtp wo_a fp8 gemv")) return 0;
    return round_out ? ds4_gpu_v41_round_bf16_tensor(low, (uint64_t)n_tok * out_all) : 1;
}
