// v41_common.metal — DeepSeek V4.1 Metal 原语的公共件(2026-10-08, Metal 侧 V4.1 落地)。
//
// 口径与 CUDA 实现(src/cuda/cuda_v41_*.inc.cu)逐式同源: f32 计算, 官方 bf16 模块边界处显式舍 bf16(v41_bf16r),
// E4M3/E8M0/FP4 的解码与 src/common/ds4_fp8.h 同一张表、同一条算式; q4_K 的 scale/min 拆包与 src/common/ds4_quantfmt.c
// 的 ds4_deq_q4_K 同式。拼接顺序: 本文件排在所有 v41_*.metal 之前(metal_source.m 的清单决定), 这里定义的函数后面的文件直接用。
//
// 出错会怎样: 这里任何一张表或一条舍入式与 src/common 漂开, 整条 V4.1 前向就静默出"一眼合理"的假值 ——
// 门是 tests/t_metal_v41.c 的逐元素金标(对着 src/common 的标量实现比)。

#define V41_SIMD 32u

// RNE 舍到 bf16 再回 f32(NaN/Inf 原样)。
inline float v41_bf16r(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    return as_type<float>(u);
}
inline ushort v41_f32_to_bf16(float x) { return (ushort)(as_type<uint>(v41_bf16r(x)) >> 16); }
inline float v41_bf16_to_f32(ushort h) { return as_type<float>((uint)h << 16); }
inline float v41_f16_to_f32(ushort h) { return float(as_type<half>(h)); }

// E4M3FN 位型 → f32(abs 0x7f = NaN, 无 Inf), 与 ds4_e4m3fn_to_f32 同式。
inline float v41_e4m3_to_f32(uint x) {
    const uint a = x & 0x7fu;
    const bool neg = (x & 0x80u) != 0u;
    if (a == 0u) return neg ? -0.0f : 0.0f;
    if (a == 0x7fu) return as_type<float>(0x7fc00000u);
    const int e = int((x >> 3) & 0x0fu), m = int(x & 0x07u);
    const float v = e == 0 ? ldexp(float(m), -9) : ldexp(1.0f + float(m) / 8.0f, e - 7);
    return neg ? -v : v;
}
// E8M0 → f32(e=0 按 0x00400000 位型), 与 ds4_e8m0_to_f32 同式。
inline float v41_e8m0_to_f32(uint e) { return as_type<float>(e == 0u ? 0x00400000u : (e << 23)); }
constant float v41_fp4_tab[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
inline float v41_fp4_to_f32(uint n) { return v41_fp4_tab[n & 15u]; }
// E4M3FN 幅值表下标 i(0..126) → 值, 与 ds4_e4m3fn_value 同式。
inline float v41_e4m3_value(int i) {
    const int e = (i >> 3) & 0x0f, m = i & 7;
    return e == 0 ? float(m) * 0.001953125f : (1.0f + float(m) * 0.125f) * ldexp(1.0f, e - 7);
}
// x → 最近的 E4M3FN 可表示值(饱和 ±448, ties 偶数尾数优先), 与 ds4_e4m3fn_round 逐式同(二分 + 相邻比)。
inline float v41_e4m3_round(float x) {
    const float sgn = x < 0.0f ? -1.0f : 1.0f, ax = min(abs(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (v41_e4m3_value(mid) <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        const float bd = abs(ax - v41_e4m3_value(best)), nd = abs(ax - v41_e4m3_value(best + 1));
        if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
    }
    return sgn * v41_e4m3_value(best);
}
inline uint v41_e4m3_f32_to_byte(float x) {
    const float v = v41_e4m3_round(x), av = abs(v);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (v41_e4m3_value(mid) <= av) lo = mid; else hi = mid - 1; }
    return (v < 0.0f && av != 0.0f) ? (uint)(lo | 0x80) : (uint)lo;
}
inline float v41_e2m1_round(float x) {
    const float sgn = x < 0.0f ? -1.0f : 1.0f, ax = min(abs(x), 6.0f);
    int best = 0; float bd = abs(ax - v41_fp4_tab[0]);
    for (int i = 1; i < 8; i++) {
        const float d = abs(ax - v41_fp4_tab[i]);
        if (d < bd || (d == bd && (i & 1) == 0 && (best & 1) != 0)) { best = i; bd = d; }
    }
    return sgn * v41_fp4_tab[best];
}
inline uint v41_fp4_f32_to_nibble(float x) {
    const float v = v41_e2m1_round(x), av = abs(v);
    uint n = 0u;
    for (int i = 1; i < 8; i++) if (av == v41_fp4_tab[i]) { n = (uint)i; break; }
    return n == 0u ? 0u : (v < 0.0f ? (n | 8u) : n);
}
inline uint v41_e8m0_f32_to_byte(float s) {
    const uint bits = as_type<uint>(s);
    return bits == 0x00400000u ? 0u : ((bits >> 23) & 0xffu);
}
// 2^ceil(log2 v): 量化缩放因子的指数部分(act_quant 与 KV 打包共用, 与 CUDA v41_pow2_ceil_log2 同式)
inline float v41_pow2_ceil_log2(float v) {
    int e; const float m = frexp(v, e);
    return ldexp(1.0f, (m == 0.5f) ? e - 1 : e);
}

// q4_K 块头 12 B scales 里第 j 组的 (scale, min), 与 ds4_quantfmt.c 的 q4k_scale_min 逐式同。
inline void v41_q4k_sm(device const uchar *sc, uint j, thread float &s, thread float &m) {
    uint a, b;
    if (j < 4u) { a = sc[j] & 63u; b = sc[j + 4u] & 63u; }
    else { a = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4); b = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4); }
    s = float(a); m = float(b);
}

// 权重类型号(与 src/metal/metal_v41_args.h 的 V41_WT_* 同值)
#define V41_WT_FP4X32 0u
#define V41_WT_Q4K    1u
#define V41_WT_BF16   2u
#define V41_WT_F32    3u
#define V41_WT_FP8BLK 4u

// 一行权重里 8 个连续元素 W[r][c0..c0+8)(c0 是 8 的倍数)解成 f32。cols = 行长(元素)。
// fp8blk 的缩放平面另给(sc + sc_off, 行块 r/32、列块 c/32, 每行 sbc 块)。
inline void v41_w8(uint wtype, device const uchar *w, ulong w_off, device const uchar *sc, ulong sc_off, uint sbc,
                   uint r, uint c0, uint cols, thread float *o) {
    if (wtype == V41_WT_FP4X32) {
        device const uchar *p = w + w_off + ((ulong)r * (cols / 32u) + (c0 / 32u)) * 17u;
        const float s = v41_e8m0_to_f32(p[16]);
        const uint q = (c0 % 32u) / 8u;
        for (uint j = 0; j < 4u; j++) {
            const uint by = p[q * 4u + j];
            o[2u * j] = v41_fp4_to_f32(by & 0xFu) * s;
            o[2u * j + 1u] = v41_fp4_to_f32(by >> 4) * s;
        }
    } else if (wtype == V41_WT_Q4K) {
        device const uchar *blk = w + w_off + ((ulong)r * (cols / 256u) + (c0 / 256u)) * 144u;
        const uint e = c0 % 256u, g = e / 64u, oo = e % 64u, hi = oo >= 32u ? 1u : 0u;
        const float d = v41_f16_to_f32((ushort)blk[0] | ((ushort)blk[1] << 8));
        const float dmin = v41_f16_to_f32((ushort)blk[2] | ((ushort)blk[3] << 8));
        float s, m; v41_q4k_sm(blk + 4, g * 2u + hi, s, m);
        const float ds = d * s, dm = dmin * m;
        device const uchar *qs = blk + 16u + g * 32u + (oo % 32u);
        for (uint j = 0; j < 8u; j++) { const uint by = qs[j]; o[j] = ds * float(hi ? (by >> 4) : (by & 0xFu)) - dm; }
    } else if (wtype == V41_WT_BF16) {
        device const ushort *p = (device const ushort *)(w + w_off + ((ulong)r * cols + c0) * 2u);
        for (uint j = 0; j < 8u; j++) o[j] = v41_bf16_to_f32(p[j]);
    } else if (wtype == V41_WT_F32) {
        device const float *p = (device const float *)(w + w_off + ((ulong)r * cols + c0) * 4u);
        for (uint j = 0; j < 8u; j++) o[j] = p[j];
    } else {   /* FP8 e4m3 + 32×32 块 ue8m0 */
        device const uchar *p = w + w_off + (ulong)r * cols + c0;
        const float s = v41_e8m0_to_f32(sc[sc_off + (ulong)(r / 32u) * sbc + (c0 / 32u)]);
        for (uint j = 0; j < 8u; j++) o[j] = v41_e4m3_to_f32(p[j]) * s;
    }
}
// 单元素 W[r][c](非热路径: 列平方和 / 转置乘的瓦片装载), 口径与 v41_w8 同一张表同一条式
inline float v41_w1(uint wtype, device const uchar *w, ulong w_off, device const uchar *sc, ulong sc_off, uint sbc, uint r, uint c, uint cols) {
    float o[8];
    v41_w8(wtype, w, w_off, sc, sc_off, sbc, r, c & ~7u, cols, o);
    return o[c & 7u];
}

// threadgroup 树形求和: 槽数 = 线程数(2 的幂, ≤1024), 结果广播给所有线程。累加形状与 CUDA 的 sh[256]/sh[1024] 树同。
inline float v41_tg_sum(float v, threadgroup float *sh, uint tid, uint nt) {
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = nt >> 1; k > 0u; k >>= 1) {
        if (tid < k) sh[tid] += sh[tid + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
inline float v41_tg_max(float v, threadgroup float *sh, uint tid, uint nt) {
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = nt >> 1; k > 0u; k >>= 1) {
        if (tid < k) sh[tid] = max(sh[tid], sh[tid + k]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
// 块内 argmax(并列取小下标; 下标 -1 = 空, 永远输), 与 CUDA v41_blk_argmax 同一棵树
inline void v41_tg_argmax(float v, int i, threadgroup float *sf, threadgroup int *si, uint tid, uint nt, thread float &ov, thread int &oi) {
    sf[tid] = v; si[tid] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = nt >> 1; k > 0u; k >>= 1) {
        if (tid < k) {
            const float bv = sf[tid + k]; const int bi = si[tid + k];
            const float av = sf[tid]; const int ai = si[tid];
            if (bi >= 0 && (ai < 0 || bv > av || (bv == av && bi < ai))) { sf[tid] = bv; si[tid] = bi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ov = sf[0]; oi = si[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// float → 可按无符号比较的 32 位键(−inf/NaN 归 0; 正数置最高位, 负数按位取反), 与 CUDA v41_topk_key 同式
inline uint v41_topk_key(float f) {
    if (!(f > -INFINITY)) return 0u;
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// 只对有限值的单调键(采样核用), 与 CUDA v41_fkey 同式
inline uint v41_fkey(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
// 有限值判定走位型: Metal 库默认 fast-math, isfinite 可能被当恒真折掉(与全仓 C 侧"-ffast-math 禁 NaN 哨兵"同一类坑)
inline bool v41_finite(float f) { return (as_type<uint>(f) & 0x7f800000u) != 0x7f800000u; }

// 公共参数块(与 src/metal/metal_v41_args.h 逐字段同序同型; 改一边必须改另一边)
struct v41_n_args { uint n0, n1, n2, n3; float f0, f1, f2, f3; };
struct v41_gemv_args {
    ulong w_off, w_gstride, sc_off, sc_gstride;
    uint in_dim, out_dim, x_stride, out_stride, ksplit, x_gstride, out_gstride, n_tok, round_out, wtype, sbc, has_skip;
};
struct v41_wgemm_args {
    ulong w_off, w_gstride, sc_off, sc_gstride;
    uint M, N, K, lda, ldc, x_gstride, out_gstride, wtype, wnn, wcols, sbc, beta, round_out, pad0;
};
struct v41_sgemm_args { uint M, N, K, lda, ldb, ldc, transA, transB; float alpha, beta; uint a_gstride, b_gstride, c_gstride, pad1; };
