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
