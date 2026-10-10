// v41_sample.metal — 设备采样核(Metal, 2026-10-08), 对应 cuda_v41_sample.inc.cu: 温度 / top-k / top-p / min-p 全在 GPU 上做,
// 一 threadgroup 一行(1024 线程), 每行 4 个 int32 落设备槽, 投机的拒绝采样同一发出。Gumbel-max + (seed, 位置, 词, 盐) 哈希的均匀数,
// 不依赖线程编排 ⇒ 同 seed 同输入必同结果。
// 与 CUDA 的两处差别(都写明): ① top_p 的定点质量用 u64 累加, Metal threadgroup 没有 64 位原子 ⇒ 拆成 lo/hi 两个 32 位原子带进位;
// ② 定点权重 CUDA 用 double 算 e·2^40, Metal 没有 double ⇒ 整数部分 2^16 档 + 小数 2^24 档拼成 u64(门槛相同量级, 只在并列边界可能差一位)。
// 必须排在 v41_common.metal 之后。

#define V41_SAMPLE_THREADS 1024u
#define V41_SAMPLE_BINS 256u

inline ulong v41_mix64(ulong x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ul; x ^= x >> 27; x *= 0x94D049BB133111EBul; x ^= x >> 31; return x;
}
// (0,1) 上的全精度均匀数(与 CUDA v41_u01 同式): 二进制档号 = 余下 41 位的前导零, 尾数取高 23 位
inline float v41_u01(ulong seed, int pos, uint i, uint salt) {
    const ulong x = v41_mix64(seed ^ (0x9E3779B97F4A7C15ul * (ulong)(uint)pos + 0xD1B54A32D192ED03ul * (ulong)i + 0x8CB92BA72F3D8DD7ul * (ulong)(salt + 1u)));
    const uint m = (uint)(x >> 41);
    const ulong r = x << 23;
    const int n = r ? (int)clz(r) : 41;
    return as_type<float>(((uint)(126 - n) << 23) | m);
}
// log1p(x) 的精确写法(Metal 没有 log1p; fast-math 的 log 在 1 附近的绝对误差会把 Gumbel 上尾打乱, 见 CUDA 文件头的三连撞)
inline float v41_log1p(float x) {
    const float w = 1.0f + x;
    float lw = precise::log(w);
    if (w != 1.0f) lw = lw * x / (w - 1.0f);
    return lw;
}
inline float v41_blk_sumf(threadgroup float *sf, float v, uint tid) { return v41_tg_sum(v, sf, tid, V41_SAMPLE_THREADS); }
// 定点权重 e·2^40 拆 (hi, lo): hi = ⌊e·2^16⌋, lo = frac·2^24
inline void v41_fix_w(float e, thread uint &hi, thread uint &lo) {
    const float s = e * 65536.0f;
    const float ip = floor(s);
    hi = (uint)ip;
    lo = (uint)((s - ip) * 16777216.0f + 0.5f);
    if (lo >= 16777216u) { lo -= 16777216u; hi += 1u; }
}
inline void v41_add64(threadgroup atomic_uint *lo, threadgroup atomic_uint *hi, uint wlo, uint whi) {
    const uint old = atomic_fetch_add_explicit(lo, wlo, memory_order_relaxed);
    const uint carry = (old + wlo < old) ? 1u : 0u;
    if (whi + carry) atomic_fetch_add_explicit(hi, whi + carry, memory_order_relaxed);
}
inline ulong v41_rd64(threadgroup atomic_uint *lo, threadgroup atomic_uint *hi) {
    return ((ulong)atomic_load_explicit(hi, memory_order_relaxed) << 32) | (ulong)atomic_load_explicit(lo, memory_order_relaxed);
}
// 基数选择: 在 {有限 且 键 ≥ floor_key} 里按键从大到小累计 weight, 首次 ≥ target 的那一项的键。mass=0: weight 1(top_k); 1: 定点概率(top_p)
inline uint v41_radix_select(device const float *l, uint V, uint floor_key, uint mass, float M, float inv_T, ulong target,
                             threadgroup atomic_uint *hlo, threadgroup atomic_uint *hhi, threadgroup uint *s_prefix, threadgroup ulong *s_target, uint tid) {
    uint prefix = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const uint mask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint b = tid; b < V41_SAMPLE_BINS; b += V41_SAMPLE_THREADS) { atomic_store_explicit(&hlo[b], 0u, memory_order_relaxed); atomic_store_explicit(&hhi[b], 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < V; i += V41_SAMPLE_THREADS) {
            const float v = l[i];
            if (!v41_finite(v)) continue;
            const uint k = v41_fkey(v);
            if (k < floor_key || (k & mask) != prefix) continue;
            uint whi = 0u, wlo = 1u;
            if (mass) v41_fix_w(precise::exp((v - M) * inv_T), whi, wlo);
            v41_add64(&hlo[(k >> shift) & 0xFFu], &hhi[(k >> shift) & 0xFFu], wlo, whi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            ulong cum = 0ul; uint sel = 0u; ulong rest = target;
            for (int b = (int)V41_SAMPLE_BINS - 1; b >= 0; b--) {
                const ulong hv = v41_rd64(&hlo[b], &hhi[b]);
                cum += hv;
                if (cum >= target) { sel = (uint)b; rest = target - (cum - hv); break; }
            }
            *s_prefix = prefix | (sel << shift); *s_target = rest;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = *s_prefix; target = *s_target;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (target == 0ul) return 0u;
    }
    return prefix;
}
// 一行的三道门: 返回键下界 cut(0 = 全保留), M = 有限最大值(Mi < 0 = 全行非有限)
inline uint v41_sample_gate(device const float *l, uint V, float inv_T, float min_p, uint top_k, float top_p,
                            threadgroup float *sf, threadgroup int *si, threadgroup atomic_uint *hlo, threadgroup atomic_uint *hhi,
                            threadgroup uint *s_prefix, threadgroup ulong *s_u64, threadgroup atomic_uint *zlo, threadgroup atomic_uint *zhi,
                            thread float &M, thread int &Mi, uint tid) {
    float best = -INFINITY; int bi = -1;
    for (uint i = tid; i < V; i += V41_SAMPLE_THREADS) { const float v = l[i]; if (v41_finite(v) && v > best) { best = v; bi = (int)i; } }
    v41_tg_argmax(best, bi, sf, si, tid, V41_SAMPLE_THREADS, M, Mi);
    if (Mi < 0) return 0u;
    uint cut = 0u;
    if (min_p > 0.0f) cut = v41_fkey(M + precise::log(min_p) / inv_T);
    uint kfloor = 0u;
    if (top_k > 0u && top_k < V) {
        kfloor = v41_radix_select(l, V, 0u, 0u, M, inv_T, (ulong)top_k, hlo, hhi, s_prefix, s_u64, tid);
        if (kfloor > cut) cut = kfloor;
    }
    if (top_p < 1.0f) {
        if (tid == 0u) { atomic_store_explicit(zlo, 0u, memory_order_relaxed); atomic_store_explicit(zhi, 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < V; i += V41_SAMPLE_THREADS) {
            const float v = l[i];
            if (!v41_finite(v) || v41_fkey(v) < kfloor) continue;
            uint whi, wlo; v41_fix_w(precise::exp((v - M) * inv_T), whi, wlo);
            v41_add64(zlo, zhi, wlo, whi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong Z = v41_rd64(zlo, zhi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* target = ⌈top_p·Z⌉: 无 double, 用 Z 的高 32 位与低 32 位分别乘(top_p 是 float, 结果取整到 u64) */
        ulong target = (ulong)ceil((float)(Z >> 32) * top_p) * 4294967296ul + (ulong)ceil((float)(Z & 0xFFFFFFFFul) * top_p);
        if (target > Z) target = Z;
        if (target == 0ul) target = 1ul;
        const uint pfloor = v41_radix_select(l, V, kfloor, 1u, M, inv_T, target, hlo, hhi, s_prefix, s_u64, tid);
        if (pfloor > cut) cut = pfloor;
    }
    return cut;
}
struct v41_sample_args { ulong seed; uint V, row0, n_rows, top_k, stream, has_q; float inv_T, min_p, top_p; uint pad0, pad1, pad2; };
kernel void kernel_v41_sample(constant v41_sample_args &a [[buffer(0)]], device int *out [[buffer(1)]], device const float *logits [[buffer(2)]],
                              device const float *qlogits [[buffer(3)]], device const int *pos [[buffer(4)]], device const int *tok [[buffer(5)]],
                              uint blk [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sf[V41_SAMPLE_THREADS]; threadgroup int si[V41_SAMPLE_THREADS];
    threadgroup atomic_uint hlo[V41_SAMPLE_BINS], hhi[V41_SAMPLE_BINS], zlo, zhi;
    threadgroup uint s_prefix; threadgroup ulong s_u64;
    const uint V = a.V, r = a.row0 + blk;
    device const float *l = logits + (ulong)r * V;
    const int p = pos[r];
    const int d = blk + 1u < a.n_rows ? tok[r + 1u] : -1;
    bool haveq = d >= 0 && a.has_q != 0u;
    device const float *lq = qlogits + (ulong)blk * V;
    const uint salt_g = 2u * a.stream, salt_a = 2u * a.stream + 1u;
    device int *o = out + 4u * blk;
    float M; int Mi;
    const uint cut = v41_sample_gate(l, V, a.inv_T, a.min_p, a.top_k, a.top_p, sf, si, hlo, hhi, &s_prefix, &s_u64, &zlo, &zhi, M, Mi, tid);
    if (Mi < 0) { if (tid == 0u) { o[0] = 0; o[1] = 0; o[2] = 0; o[3] = 0; } return; }
    float Mq = 0.0f; int Mqi = -1; uint cutq = 0u; float Zp = 1.0f, Zq = 1.0f;
    if (haveq) {
        cutq = v41_sample_gate(lq, V, a.inv_T, a.min_p, a.top_k, a.top_p, sf, si, hlo, hhi, &s_prefix, &s_u64, &zlo, &zhi, Mq, Mqi, tid);
        if (Mqi < 0) haveq = false;
    }
    if (haveq) {
        float zp = 0.0f, zq = 0.0f;
        for (uint i = tid; i < V; i += V41_SAMPLE_THREADS) {
            const float v = l[i], vq = lq[i];
            if (v41_finite(v) && v41_fkey(v) >= cut) zp += precise::exp((v - M) * a.inv_T);
            if (v41_finite(vq) && v41_fkey(vq) >= cutq) zq += precise::exp((vq - Mq) * a.inv_T);
        }
        Zp = v41_blk_sumf(sf, zp, tid); Zq = v41_blk_sumf(sf, zq, tid);
    }
    float gb = -INFINITY, gxb = -INFINITY, zk = 0.0f, md = 0.0f, mqd = 0.0f; int gi = -1, gxi = -1;
    for (uint i = tid; i < V; i += V41_SAMPLE_THREADS) {
        const float v = l[i];
        if (haveq && (int)i == d) { const float vq = lq[i]; mqd = (v41_finite(vq) && v41_fkey(vq) >= cutq) ? precise::exp((vq - Mq) * a.inv_T) : 0.0f; }
        if (!v41_finite(v) || v41_fkey(v) < cut) continue;
        const float m = precise::exp((v - M) * a.inv_T);
        zk += m;
        if ((int)i == d) md = m;
        const float u = v41_u01(a.seed, p, i, salt_g);
        const float g = -precise::log(-v41_log1p(-u));
        const float key = v * a.inv_T + g;
        if (key > gb) { gb = key; gi = (int)i; }
        if (haveq) {
            const float vq = lq[i];
            const float qm = (v41_finite(vq) && v41_fkey(vq) >= cutq) ? precise::exp((vq - Mq) * a.inv_T) : 0.0f;
            const float rres = m / Zp - qm / Zq;
            if (rres > 0.0f) { const float rk = precise::log(rres) + g; if (rk > gxb) { gxb = rk; gxi = (int)i; } }
        } else if ((int)i != d && key > gxb) { gxb = key; gxi = (int)i; }
    }
    float tv; int full, resid;
    v41_tg_argmax(gb, gi, sf, si, tid, V41_SAMPLE_THREADS, tv, full);
    v41_tg_argmax(gxb, gxi, sf, si, tid, V41_SAMPLE_THREADS, tv, resid);
    const float ZK = v41_blk_sumf(sf, zk, tid), MD = v41_blk_sumf(sf, md, tid), MQD = v41_blk_sumf(sf, mqd, tid);
    if (tid == 0u) {
        int acc = 0;
        if (d >= 0) {
            const float pd = MD / ZK;
            if (pd <= 0.0f) acc = 0;
            else if (resid < 0) acc = 1;
            else if (haveq) { const float qd = MQD / Zq; acc = (qd <= 0.0f || v41_u01(a.seed, p, (uint)d, salt_a) < pd / qd) ? 1 : 0; }
            else acc = v41_u01(a.seed, p, (uint)d, salt_a) < pd ? 1 : 0;
        }
        o[0] = full; o[1] = acc; o[2] = resid >= 0 ? resid : full; o[3] = Mi;   // [3] = 原始 argmax, 用途见 CUDA 版头注释
    }
}
