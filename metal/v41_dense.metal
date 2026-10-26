// v41_dense.metal — DeepSeek V4.1 稠密线性层(Metal, 2026-10-08): 解码小批 GEMV / 预填解量化 GEMM / f32 GEMM / 嵌入 / 小件。
// 对应 CUDA: cuda_v41_1 / cuda_v41_4 / cuda_v41_q4k / cuda_v41_gemv_highprec。口径同: 权重精确值 × bf16 格点激活, f32 累加,
// round_out 时出口舍 bf16。CUDA 的 cuBLAS 路在这里是自写的 simdgroup 8×8 瓦片 GEMM(权重边解边乘, 不落 bf16 暂存)。
// 必须排在 v41_common.metal 之后。

// ---- 解码小批 GEMV(n ≤ 8): 一 threadgroup 8 个 simdgroup; 一行由 ksplit 个 simdgroup 分 K 段, 每段 256 元素一轮(lane 管 8 个连续元素),
// 段和经 threadgroup 按 k 升序相加(确定序)。grid (ceil(out_dim/rpb), n_groups); n_groups>1 = 块对角(wo_a), 权重/输入/输出按组步长偏移。
template <uint NT>
kernel void kernel_v41_gemv(constant v41_gemv_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                            device float *out [[buffer(3)]], device const int *skip [[buffer(4)]], device const uchar *sc [[buffer(5)]],
                            uint3 tgpig [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8][8];
    if (a.has_skip != 0u && skip[0] >= 0) return;   /* markov 偏置缓存命中: 整网格直接退 */
    const uint g = tgpig.y;
    const ulong w_off = a.w_off + (ulong)g * a.w_gstride, sc_off = a.sc_off + (ulong)g * a.sc_gstride;
    x += (ulong)g * a.x_gstride; out += (ulong)g * a.out_gstride;
    const uint rpb = 8u / a.ksplit, rloc = sgitg / a.ksplit, kpart = sgitg % a.ksplit;
    const uint r = tgpig.x * rpb + rloc;
    float acc[NT];
    for (uint t = 0; t < NT; t++) acc[t] = 0.0f;
    if (r < a.out_dim) {
        const uint nchunk = (a.in_dim + 255u) / 256u;
        for (uint ch = kpart; ch < nchunk; ch += a.ksplit) {
            const uint c0 = ch * 256u + lane * 8u;
            if (c0 < a.in_dim) {
                float wv[8];
                v41_w8(a.wtype, w, w_off, sc, sc_off, a.sbc, r, c0, a.in_dim, wv);
                for (uint t = 0; t < NT; t++) {
                    device const float *xt = x + (ulong)t * a.x_stride + c0;
                    /* 八条独立累加(与 CUDA 同: 合成一句会被 fast-math 改成加法树, 累加序变) */
                    acc[t] += wv[0] * xt[0]; acc[t] += wv[1] * xt[1]; acc[t] += wv[2] * xt[2]; acc[t] += wv[3] * xt[3];
                    acc[t] += wv[4] * xt[4]; acc[t] += wv[5] * xt[5]; acc[t] += wv[6] * xt[6]; acc[t] += wv[7] * xt[7];
                }
            }
        }
        for (uint t = 0; t < NT; t++) acc[t] = simd_sum(acc[t]);
    }
    if (a.ksplit > 1u) {
        if (lane == 0u) for (uint t = 0; t < NT; t++) red[sgitg][t] = acc[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kpart == 0u && lane == 0u && r < a.out_dim) {
            for (uint t = 0; t < NT; t++) {
                float v = 0.0f;
                for (uint k = 0; k < a.ksplit; k++) v += red[rloc * a.ksplit + k][t];
                out[(ulong)t * a.out_stride + r] = a.round_out ? v41_bf16r(v) : v;
            }
        }
    } else if (lane == 0u && r < a.out_dim) {
        for (uint t = 0; t < NT; t++) out[(ulong)t * a.out_stride + r] = a.round_out ? v41_bf16r(acc[t]) : acc[t];
    }
}
typedef decltype(kernel_v41_gemv<1>) v41_gemv_t;
template [[host_name("kernel_v41_gemv_nt1")]] kernel v41_gemv_t kernel_v41_gemv<1>;
template [[host_name("kernel_v41_gemv_nt2")]] kernel v41_gemv_t kernel_v41_gemv<2>;
template [[host_name("kernel_v41_gemv_nt3")]] kernel v41_gemv_t kernel_v41_gemv<3>;
template [[host_name("kernel_v41_gemv_nt4")]] kernel v41_gemv_t kernel_v41_gemv<4>;
template [[host_name("kernel_v41_gemv_nt5")]] kernel v41_gemv_t kernel_v41_gemv<5>;
template [[host_name("kernel_v41_gemv_nt6")]] kernel v41_gemv_t kernel_v41_gemv<6>;
template [[host_name("kernel_v41_gemv_nt7")]] kernel v41_gemv_t kernel_v41_gemv<7>;
template [[host_name("kernel_v41_gemv_nt8")]] kernel v41_gemv_t kernel_v41_gemv<8>;

// ---- q4_K GEMV: each token runs identical arithmetic (grid.z selects token).
// 移植自 llama.cpp kernel_mul_mv_q4_K_f32_impl(MIT, ggml-org/llama.cpp)。
// 每 simdgroup 2 行, 每 threadgroup 2 个 simdgroup(64 线程); 一 lane 的 32 个 x 读进寄存器后两行复用; 16 位读量化字节 + 掩码不移位;
// min 项按块 Σx 提出来。累加序与 kernel_v41_gemv<1> 不同 ⇒ 门是 NLL。要求 out_dim % 4 == 0、in_dim % 256 == 0(主机校验)。
#define V41_Q4K_NR0 2
#define V41_Q4K_NSG 2
kernel void kernel_v41_gemv_q4k_ll(constant v41_gemv_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                                   device float *out [[buffer(3)]], device const int *skip [[buffer(4)]],
                                   uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint t = tgpig.z;
    if (a.has_skip != 0u && skip[t] >= 0) return;
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const uint g = tgpig.y;
    x += (ulong)t * a.x_stride + (ulong)g * a.x_gstride;
    out += (ulong)t * a.out_stride + (ulong)g * a.out_gstride;
    const short ix = tiisg / 8, it = tiisg % 8, iq = it / 4, ir = it % 4;
    const uint nb = a.in_dim / 256u, rowb = nb * 144u;
    const uint first_row = (tgpig.x * V41_Q4K_NSG + sgitg) * V41_Q4K_NR0;
    device const uchar *wrow = w + a.w_off + (ulong)g * a.w_gstride + (ulong)first_row * rowb;
    float yl[16], yh[16];
    float sumf[V41_Q4K_NR0] = { 0.f, 0.f };
    device const float *y4 = x + ix * 256 + 64 * iq + 8 * ir;
    ushort sc16[4];
    thread const uchar *sc8 = (thread const uchar *)sc16;
    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = { 0.f, 0.f, 0.f, 0.f };
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        device const uchar *blk = wrow + (ulong)ib * 144u;
        device const ushort *sc = (device const ushort *)(blk + 4) + iq;
        device const ushort *q1 = (device const ushort *)(blk + 16) + 16 * iq + 4 * ir;
        device const half *dh = (device const half *)blk;
        for (short row = 0; row < V41_Q4K_NR0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort *q2 = q1 + 32;
            float4 acc1 = { 0.f, 0.f, 0.f, 0.f }, acc2 = { 0.f, 0.f, 0.f, 0.f };
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += float(dh[0]) * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                         (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                         (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                         (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += rowb / 2; sc += rowb / 2; dh += rowb / 2;
        }
        y4 += 4 * 256;
    }
    for (short row = 0; row < V41_Q4K_NR0; ++row) {
        const uint r = first_row + row;
        const float sa = simd_sum(sumf[row]);
        if (tiisg == 0 && r < a.out_dim) out[r] = a.round_out ? v41_bf16r(sa) : sa;
    }
}

// ---- q4_K、2 ≤ n_tok ≤ 8 的验证批 GEMV(2026-10-18 M5, DSpark 验证路): 一 simdgroup 管 8 行、32 lane 同时啃同一个 q4_K 块
// (lane/8 = 行对, lane%8 = 块内 32 个权重, 布局同 ll 核), 4 个行对读同一段 x ⇒ 硬件合并成一次广播; 每行权重整解成 f32 后对 NT 个 token 各一次 FMA。
// 一 threadgroup 4 个 simdgroup 管同 8 行、按块轮流分 K, 末尾 threadgroup 求和(小矩阵也有足够的并行)。累加序与 nt 核不同 ⇒ 门是 NLL。
// 要求 out_dim % 8 == 0、in_dim % 256 == 0、x 行 16 B 对齐(主机校验)。
template <uint NT>
kernel void kernel_v41_gemv_q4k_w(constant v41_gemv_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                                  device float *out [[buffer(3)]], device const int *skip [[buffer(4)]],
                                  uint3 tgpig [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[4][8 * NT];
    if (a.has_skip != 0u && skip[0] >= 0) return;
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const uint g = tgpig.y;
    x += (ulong)g * a.x_gstride; out += (ulong)g * a.out_gstride;
    const short rp = tiisg / 8, it = tiisg % 8, iq = it / 4, ir = it % 4;
    const uint nb = a.in_dim / 256u, rowb = nb * 144u;
    const uint first_row = tgpig.x * 8u + (uint)rp * 2u;
    device const uchar *wrow = w + a.w_off + (ulong)g * a.w_gstride + (ulong)first_row * rowb;
    float sumf[NT][2];
    for (uint t = 0; t < NT; t++) { sumf[t][0] = 0.f; sumf[t][1] = 0.f; }
    ushort sc16[4];
    thread const uchar *sc8 = (thread const uchar *)sc16;
    for (uint ib = sgitg; ib < nb; ib += 4u) {
        float4 wv[2][8];
        for (short row = 0; row < 2; row++) {
            device const uchar *blk = wrow + (ulong)row * rowb + (ulong)ib * 144u;
            device const ushort *sc = (device const ushort *)(blk + 4) + iq;
            const half2 dd = *(device const half2 *)blk;
            const float d = float(dd.x), dm = float(dd.y);
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            const uint2 b1 = *(device const uint2 *)(blk + 16u + 32u * iq + 8u * ir);
            const uint2 b2 = *(device const uint2 *)(blk + 16u + 32u * iq + 8u * ir + 64u);
            const float s0 = d * sc8[0], s1 = d * sc8[1], s4 = d * sc8[4], s5 = d * sc8[5];
            const float m0 = dm * sc8[2], m1 = dm * sc8[3], m4 = dm * sc8[6], m5 = dm * sc8[7];
            wv[row][0] = s0 * float4(uint4(b1.x, b1.x >> 8, b1.x >> 16, b1.x >> 24) & 0xFu) - m0;
            wv[row][1] = s0 * float4(uint4(b1.y, b1.y >> 8, b1.y >> 16, b1.y >> 24) & 0xFu) - m0;
            wv[row][2] = s1 * float4(uint4(b1.x >> 4, b1.x >> 12, b1.x >> 20, b1.x >> 28) & 0xFu) - m1;
            wv[row][3] = s1 * float4(uint4(b1.y >> 4, b1.y >> 12, b1.y >> 20, b1.y >> 28) & 0xFu) - m1;
            wv[row][4] = s4 * float4(uint4(b2.x, b2.x >> 8, b2.x >> 16, b2.x >> 24) & 0xFu) - m4;
            wv[row][5] = s4 * float4(uint4(b2.y, b2.y >> 8, b2.y >> 16, b2.y >> 24) & 0xFu) - m4;
            wv[row][6] = s5 * float4(uint4(b2.x >> 4, b2.x >> 12, b2.x >> 20, b2.x >> 28) & 0xFu) - m5;
            wv[row][7] = s5 * float4(uint4(b2.y >> 4, b2.y >> 12, b2.y >> 20, b2.y >> 28) & 0xFu) - m5;
        }
        for (uint t = 0; t < NT; t++) {
            device const float4 *y4 = (device const float4 *)(x + (ulong)t * a.x_stride + ib * 256u + 64u * iq + 8u * ir);
            const float4 y0 = y4[0], y1 = y4[1], y2 = y4[8], y3 = y4[9], y4a = y4[32], y5 = y4[33], y6 = y4[40], y7 = y4[41];
            for (short row = 0; row < 2; row++) {
                sumf[t][row] += (dot(wv[row][0], y0) + dot(wv[row][1], y1)) + (dot(wv[row][2], y2) + dot(wv[row][3], y3)) +
                                (dot(wv[row][4], y4a) + dot(wv[row][5], y5)) + (dot(wv[row][6], y6) + dot(wv[row][7], y7));
            }
        }
    }
    /* 行对内 8 个 lane 求和(xor 4/2/1 不跨行对), 再跨 4 个 simdgroup 求和 */
    for (uint t = 0; t < NT; t++) for (short row = 0; row < 2; row++) {
        float v = sumf[t][row];
        v += simd_shuffle_xor(v, 4u); v += simd_shuffle_xor(v, 2u); v += simd_shuffle_xor(v, 1u);
        if (it == 0) red[sgitg][((uint)rp * 2u + (uint)row) * NT + t] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = (uint)sgitg * 32u + tiisg;
    if (tid < 8u * NT) {
        const uint rl = tid / NT, t = tid % NT, r = tgpig.x * 8u + rl;
        const float v = (red[0][tid] + red[1][tid]) + (red[2][tid] + red[3][tid]);
        if (r < a.out_dim) out[(ulong)t * a.out_stride + r] = a.round_out ? v41_bf16r(v) : v;
    }
}
typedef decltype(kernel_v41_gemv_q4k_w<2>) v41_q4k_w_t;
template [[host_name("kernel_v41_gemv_q4k_w2")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<2>;
template [[host_name("kernel_v41_gemv_q4k_w3")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<3>;
template [[host_name("kernel_v41_gemv_q4k_w4")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<4>;
template [[host_name("kernel_v41_gemv_q4k_w5")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<5>;
template [[host_name("kernel_v41_gemv_q4k_w6")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<6>;
template [[host_name("kernel_v41_gemv_q4k_w7")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<7>;
template [[host_name("kernel_v41_gemv_q4k_w8")]] kernel v41_q4k_w_t kernel_v41_gemv_q4k_w<8>;

// ---- 预填 GEMM(权重边解边乘): out[M][N] (+)= x[M][K] · B, B(k,n) = wnn ? W[k][n] : W[n][k]。
// 瓦片 64×64, 128 线程 = 4 个 simdgroup 排 2×2(每个 32×32 = 4×4 个 8×8), K 步 16。grid (ceil(N/64), ceil(M/64), n_groups)。
// 要求 K、N(wnn 时)是 8 的倍数(v41_w8 一次解 8 个), 调用方校验。
kernel void kernel_v41_wgemm(constant v41_wgemm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                             device float *out [[buffer(3)]], device const uchar *sc [[buffer(4)]],
                             uint3 tgpig [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[64 * 16], Bs[16 * 64], Cs[64 * 64];
    const uint g = tgpig.z;
    const ulong w_off = a.w_off + (ulong)g * a.w_gstride, sc_off = a.sc_off + (ulong)g * a.sc_gstride;
    x += (ulong)g * a.x_gstride; out += (ulong)g * a.out_gstride;
    const uint m0 = tgpig.y * 64u, n0 = tgpig.x * 64u, sm = (sgitg >> 1) * 32u, sn = (sgitg & 1u) * 32u;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4u; i++) for (uint j = 0; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < a.K; k0 += 16u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   /* A 瓦片: 线程 → 行 tid/2, 8 个连续 k */
            const uint m = m0 + (tid >> 1), kk = k0 + (tid & 1u) * 8u;
            for (uint j = 0; j < 8u; j++) As[(tid >> 1) * 16u + (tid & 1u) * 8u + j] = (m < a.M && kk + j < a.K) ? x[(ulong)m * a.lda + kk + j] : 0.0f;
        }
        if (a.wnn == 0u) {   /* B(k,n) = W[n][k]: 线程 → n = tid/2, 8 个连续 k */
            const uint n = n0 + (tid >> 1), kk = k0 + (tid & 1u) * 8u;
            float o[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (n < a.N && kk < a.K) v41_w8(a.wtype, w, w_off, sc, sc_off, a.sbc, n, kk, a.wcols, o);
            for (uint j = 0; j < 8u; j++) Bs[((tid & 1u) * 8u + j) * 64u + (tid >> 1)] = (kk + j < a.K) ? o[j] : 0.0f;
        } else {             /* B(k,n) = W[k][n]: 线程 → k = tid/8, 8 个连续 n */
            const uint k = k0 + (tid >> 3), nn = n0 + (tid & 7u) * 8u;
            float o[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (k < a.K && nn < a.N) v41_w8(a.wtype, w, w_off, sc, sc_off, a.sbc, k, nn, a.wcols, o);
            for (uint j = 0; j < 8u; j++) Bs[(tid >> 3) * 64u + (tid & 7u) * 8u + j] = (nn + j < a.N) ? o[j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 16u; kk += 8u) {
            simdgroup_float8x8 ma[4], mb[4];
            for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], As + (sm + 8u * i) * 16u + kk, 16);
            for (uint j = 0; j < 4u; j++) simdgroup_load(mb[j], Bs + kk * 64u + sn + 8u * j, 64);
            for (uint i = 0; i < 4u; i++) for (uint j = 0; j < 4u; j++) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4u; i++) for (uint j = 0; j < 4u; j++) simdgroup_store(acc[i][j], Cs + (sm + 8u * i) * 64u + sn + 8u * j, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 64u * 64u; e += 128u) {
        const uint mi = e >> 6, ni = e & 63u, m = m0 + mi, n = n0 + ni;
        if (m < a.M && n < a.N) {
            float v = Cs[e];
            if (a.beta) v += out[(ulong)m * a.ldc + n];
            out[(ulong)m * a.ldc + n] = a.round_out ? v41_bf16r(v) : v;
        }
    }
}

// ---- f32 GEMM: C[M][N] = alpha·op(A)·op(B) + beta·C(行主序; transA: A 存 [K][M], transB: B 存 [N][K])。瓦片同上。
kernel void kernel_v41_sgemm(constant v41_sgemm_args &a [[buffer(0)]], device const float *A [[buffer(1)]], device const float *B [[buffer(2)]],
                             device float *C [[buffer(3)]], uint3 tgpig [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[64 * 16], Bs[16 * 64], Cs[64 * 64];
    const uint g = tgpig.z;
    A += (ulong)g * a.a_gstride; B += (ulong)g * a.b_gstride; C += (ulong)g * a.c_gstride;
    const uint m0 = tgpig.y * 64u, n0 = tgpig.x * 64u, sm = (sgitg >> 1) * 32u, sn = (sgitg & 1u) * 32u;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4u; i++) for (uint j = 0; j < 4u; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < a.K; k0 += 16u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 64u * 16u; e += 128u) {   /* A 瓦片 [m][k] */
            const uint mi = e >> 4, ki = e & 15u, m = m0 + mi, k = k0 + ki;
            float v = 0.0f;
            if (m < a.M && k < a.K) v = a.transA ? A[(ulong)k * a.lda + m] : A[(ulong)m * a.lda + k];
            As[e] = v;
        }
        for (uint e = tid; e < 16u * 64u; e += 128u) {   /* B 瓦片 [k][n] */
            const uint ki = e >> 6, ni = e & 63u, k = k0 + ki, n = n0 + ni;
            float v = 0.0f;
            if (k < a.K && n < a.N) v = a.transB ? B[(ulong)n * a.ldb + k] : B[(ulong)k * a.ldb + n];
            Bs[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 16u; kk += 8u) {
            simdgroup_float8x8 ma[4], mb[4];
            for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], As + (sm + 8u * i) * 16u + kk, 16);
            for (uint j = 0; j < 4u; j++) simdgroup_load(mb[j], Bs + kk * 64u + sn + 8u * j, 64);
            for (uint i = 0; i < 4u; i++) for (uint j = 0; j < 4u; j++) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4u; i++) for (uint j = 0; j < 4u; j++) simdgroup_store(acc[i][j], Cs + (sm + 8u * i) * 64u + sn + 8u * j, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 64u * 64u; e += 128u) {
        const uint mi = e >> 6, ni = e & 63u, m = m0 + mi, n = n0 + ni;
        if (m < a.M && n < a.N) {
            const ulong o = (ulong)m * a.ldc + n;
            C[o] = a.alpha * Cs[e] + (a.beta != 0.0f ? a.beta * C[o] : 0.0f);
        }
    }
}

// ---- 嵌入取行: out[t][dim] = W[tok[t]](fp4x32 舍 bf16 / q4_K 原值, 与 CUDA 两核同口径)。grid (dim/8, n_tok), 一线程 8 个元素。
kernel void kernel_v41_embed(constant v41_n_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const int *tok [[buffer(2)]],
                             device float *out [[buffer(3)]], constant ulong &w_off [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint n_vocab = a.n0, dim = a.n1, wtype = a.n2, t = gid.y, c0 = gid.x * 8u;
    if (c0 >= dim) return;
    int id = tok[t];
    if (wtype == V41_WT_FP4X32) { if (id < 0 || (uint)id >= n_vocab) id = 0; }
    else if (id < 0 || (uint)id >= n_vocab) return;   /* q4_K 版: 越界 id 不写(与 CUDA 同) */
    float o[8];
    v41_w8(wtype, w, w_off, w, 0ul, 0u, (uint)id, c0, dim, o);
    for (uint j = 0; j < 8u; j++) out[(ulong)t * dim + c0 + j] = wtype == V41_WT_FP4X32 ? v41_bf16r(o[j]) : o[j];
}

// ---- 小件 ----
kernel void kernel_v41_round_bf16(device float *x [[buffer(0)]], constant ulong &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) x[i] = v41_bf16r(x[i]);
}
kernel void kernel_v41_scale_round(device float *x [[buffer(0)]], constant ulong &n [[buffer(1)]], constant float &s [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) x[i] = v41_bf16r(x[i] * s);
}
kernel void kernel_v41_add(device float *a [[buffer(0)]], device const float *b [[buffer(1)]], constant ulong &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) a[i] += b[i];
}
kernel void kernel_v41_axpy(device float *y [[buffer(0)]], device const float *x [[buffer(1)]], constant ulong &n [[buffer(2)]], constant float &al [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) y[i] += al * x[i];
}
kernel void kernel_v41_fill_u32(device uint *p [[buffer(0)]], constant ulong &n [[buffer(1)]], constant uint &v [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) p[i] = v;
}
// 行加: dst 第 row 行 += src(markov 偏置); row 偏移由主机折进 setBuffer offset, 这里就是一维加
kernel void kernel_v41_expand_hc(constant v41_n_args &a [[buffer(0)]], device float *hc [[buffer(1)]], device const float *x [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint n_embd = a.n0, n_hc = a.n1, d = gid.x, n = gid.y;
    if (d >= n_embd) return;
    const float v = x[(ulong)n * n_embd + d];
    for (uint c = 0; c < n_hc; c++) hc[((ulong)n * n_hc + c) * n_embd + d] = v;
}
// argmax(同值取小下标), 一 threadgroup 1024 线程一行
kernel void kernel_v41_argmax(device int *idx [[buffer(0)]], device const float *row [[buffer(1)]], constant uint &n [[buffer(2)]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sf[1024]; threadgroup int si[1024];
    float best = -INFINITY; int bi = 0;
    for (uint i = tid; i < n; i += 1024u) { const float v = row[i]; if (v > best) { best = v; bi = (int)i; } }
    float ov; int oi;
    v41_tg_argmax(best, bi, sf, si, tid, 1024u, ov, oi);
    if (tid == 0u) idx[0] = oi;
}
// 出口头列平方和 Σ_v W[v][d]² → out[d](fp4x32 / q4_K 同一入口): 一 threadgroup 一列, 256 线程沿 v 跨步
kernel void kernel_v41_colnorm(constant v41_n_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device float *out [[buffer(2)]],
                               constant ulong &w_off [[buffer(3)]], uint d [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    const uint V = a.n0, D = a.n1, wtype = a.n2;
    float s = 0.0f;
    for (uint v = tid; v < V; v += 256u) { const float wv = v41_w1(wtype, w, w_off, w, 0ul, 0u, v, d, D); s += wv * wv; }
    const float r = v41_tg_sum(s, sh, tid, 256u);
    if (tid == 0u) out[d] = r;
}
// engram 查表行 dequant: raw[n_rows][hd + hd/32](fp8 e4m3 + 每 32 维一个 ue8m0) → bf16 格点 f32
kernel void kernel_v41_engram_rows(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const uchar *raw [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    const uint n_rows = a.n0, hd = a.n1;
    if (i >= n_rows * hd) return;
    const uint r = i / hd, d = i % hd, stride = hd + hd / 32u;
    device const uchar *row = raw + (ulong)r * stride;
    out[i] = v41_bf16r(v41_e4m3_to_f32(row[d]) * v41_e8m0_to_f32(row[hd + d / 32u]));
}
// 按设备上的 ids[which] 取表的一行(bf16/f32 由 elem_bytes 定)→ out 第 out_row 行
kernel void kernel_v41_row_gather(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const uchar *tab [[buffer(2)]],
                                  device const int *ids [[buffer(3)]], constant ulong &tab_off [[buffer(4)]], uint d [[thread_position_in_grid]]) {
    const uint which = a.n0, dim = a.n1, out_row = a.n2, is_f32 = a.n3;
    if (d >= dim) return;
    const int id = ids[which];
    float v = 0.0f;
    if (id >= 0) {
        if (is_f32) v = ((device const float *)(tab + tab_off))[(ulong)id * dim + d];
        else v = v41_bf16_to_f32(((device const ushort *)(tab + tab_off))[(ulong)id * dim + d]);
    }
    out[(ulong)out_row * dim + d] = v;
}
// 批量取行(草稿蒸馏): out[r][dim] = tab[ids[r]][dim]; 越界/负 id 写 0
kernel void kernel_v41_rows_gather(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const uchar *tab [[buffer(2)]],
                                   device const int *ids [[buffer(3)]], constant ulong &tab_off [[buffer(4)]], constant ulong &n_rows_tab [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    const uint dim = a.n1, is_f32 = a.n3, d = gid.x, r = gid.y;
    if (d >= dim) return;
    const int id = ids[r];
    float v = 0.0f;
    if (id >= 0 && (ulong)id < n_rows_tab) {
        if (is_f32) v = ((device const float *)(tab + tab_off))[(ulong)id * dim + d];
        else v = v41_bf16_to_f32(((device const ushort *)(tab + tab_off))[(ulong)id * dim + d]);
    }
    out[(ulong)r * dim + d] = v;
}
// markov 偏置缓存: lookup(一 threadgroup, 槽号 = 线程号) / add(命中 += cache[槽], 否则 += bias 并存槽)
kernel void kernel_v41_mkcache_lookup(constant v41_n_args &a [[buffer(0)]], device int *hit [[buffer(1)]], device int *cache_ids [[buffer(2)]],
                                      device uint *next [[buffer(3)]], device const int *ids [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int found;
    const uint which = a.n0, n_slots = a.n1;
    if (tid == 0u) found = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int id = ids[which];
    if (tid < n_slots && cache_ids[tid] == id) found = (int)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        if (found >= 0) { hit[0] = found; return; }
        const uint s = next[0] % n_slots;
        next[0] = s + 1u;
        cache_ids[s] = id;
        hit[0] = -(int)s - 2;
    }
}
kernel void kernel_v41_mkcache_add(device float *dst [[buffer(0)]], device const float *bias [[buffer(1)]], device float *cache [[buffer(2)]],
                                   device const int *hit [[buffer(3)]], constant ulong &n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= n) return;
    const int h = hit[0];
    if (h >= 0) dst[i] += cache[(ulong)h * n + i];
    else { const float b = bias[i]; dst[i] += b; cache[(ulong)(-h - 2) * n + i] = b; }
}
// 环 → 连续: dst[t] = ring[(first + t) % cap]; firstd 非空则起始行从设备读
kernel void kernel_v41_ring_rows(constant v41_n_args &a [[buffer(0)]], device float *dst [[buffer(1)]], device const float *ring [[buffer(2)]],
                                 device const int *firstd [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint row_floats = a.n0, cap = a.n1, has_firstd = a.n3;
    uint first_row = a.n2;
    if (has_firstd) first_row = (uint)firstd[0];
    device const float *src = ring + (ulong)((first_row + t) % cap) * row_floats;
    device float *d = dst + (ulong)t * row_floats;
    for (uint j = tid; j < row_floats; j += 256u) d[j] = src[j];
}
// hc 四路均值 → main_hidden 环的第 slot 段: 第 t 行落 ((pos + t) % cap) 格
struct v41_hcmean_args { uint E, n_hc, n_rows, src_row0, slot, n_slot, dst_pos0, cap, has_posd, pad; };
kernel void kernel_v41_hc_mean(constant v41_hcmean_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *hc [[buffer(2)]],
                               device const int *posd [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n_rows * a.E) return;
    const uint t = i / a.E, d = i % a.E;
    device const float *h = hc + (ulong)(a.src_row0 + t) * a.n_hc * a.E + d;
    float s = 0.0f;
    for (uint c = 0; c < a.n_hc; c++) s += h[(ulong)c * a.E];
    const uint pos = a.has_posd ? (uint)posd[0] + a.src_row0 : a.dst_pos0;
    out[(ulong)((pos + t) % a.cap) * a.n_slot * a.E + (ulong)a.slot * a.E + d] = s / float(a.n_hc);
}
// 低秩件小批应用(草稿态): T[n][K] = x·Bᵀ(一 simdgroup 一个 (行, k)); y += T·A(一线程一个 (行, d))
kernel void kernel_v41_lowrank_t(constant v41_n_args &a [[buffer(0)]], device float *T [[buffer(1)]], device const float *x [[buffer(2)]],
                                 device const float *B [[buffer(3)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint D = a.n0, K = a.n1, nw = a.n2, gw = gid >> 5, r = gw / K, k = gw % K;
    if (gw >= nw) return;
    device const float *xr = x + (ulong)r * D, *br = B + (ulong)k * D;
    float s = 0.0f;
    for (uint d = lane; d < D; d += 32u) s += xr[d] * br[d];
    s = simd_sum(s);
    if (lane == 0u) T[(ulong)r * K + k] = s;
}
kernel void kernel_v41_lowrank_y(constant v41_n_args &a [[buffer(0)]], device float *y [[buffer(1)]], device const float *T [[buffer(2)]],
                                 device const float *A [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint D = a.n0, K = a.n1, d = gid.x, r = gid.y;
    if (d >= D) return;
    device const float *tr = T + (ulong)r * K;
    float s = 0.0f;
    for (uint k = 0; k < K; k++) s += tr[k] * A[(ulong)k * D + d];
    y[(ulong)r * D + d] += s;
}
