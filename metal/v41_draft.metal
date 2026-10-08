// v41_draft.metal — 草稿器蒸馏原语(Metal, 2026-10-08), 对应 cuda_draft_attn.inc.cu: 批量全可见块注意力前向(出 lse)/反向、陪审团、总变差损失、
// markov 表取行与梯度散加。块 b 的键 = 块前面 nh = min(window, bpos[b]) 个历史行(hist 第 bpos[b]−nh+kk−hbase 行) + 块内 B 行, 块内全可见。
// 与 CUDA 的差别: 陪审团/总变差的 Σexp 在 CUDA 用 double, Metal 没有 double ⇒ f32 两遍(先 max 再和)。必须排在 v41_common.metal 之后。

struct v41_dk_args { uint hbase, B, window, n_head, hd, nb, pad0, pad1; float scale; float pad2, pad3, pad4; };
inline device const float *v41_dk_key_row(device const float *hist, uint hbase, device const float *blk, uint b, uint ib, uint nh, uint B, uint kk, uint hd) {
    return kk < nh ? hist + (ulong)(ib - nh + kk - hbase) * hd : blk + ((ulong)b * B + (kk - nh)) * hd;
}
// 前向: 一 threadgroup 一 (行, 8 头), 8 键一片; 两遍(先 max/sum, 再 P·V), 出 o(bf16 格点)与 lse = m + log(den)(含 sink)
kernel void kernel_v41_draft_attn_fwd(constant v41_dk_args &a [[buffer(0)]], device float *o [[buffer(1)]], device float *lse [[buffer(2)]],
                                      device const float *q [[buffer(3)]], device const float *hist [[buffer(4)]], device const float *blk [[buffer(5)]],
                                      device const int *bpos [[buffer(6)]], device const float *sink [[buffer(7)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ks[8][512];
    const uint r = tg.x, h = tg.y * 8u + sgitg, hd = a.hd, per = hd / 32u, b = r / a.B;
    const uint ib = (uint)bpos[b], nh = ib < a.window ? ib : a.window, nkeys = nh + a.B;
    float qa[16], acc[16];
    for (uint e = 0; e < per; e++) { qa[e] = v41_bf16r(q[((ulong)r * a.n_head + h) * hd + lane * per + e]); acc[e] = 0.0f; }
    float mx = -1e30f, sum = 0.0f;
    for (uint pass = 0; pass < 2u; pass++) {
        for (uint base = 0; base < nkeys; base += 8u) {
            const uint nt = (nkeys - base) < 8u ? (nkeys - base) : 8u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgitg < nt) {
                device const float *krow = v41_dk_key_row(hist, a.hbase, blk, b, ib, nh, a.B, base + sgitg, hd);
                for (uint d = lane; d < hd; d += 32u) ks[sgitg][d] = v41_bf16r(krow[d]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = 0; t < nt; t++) {
                float s = 0.0f;
                for (uint e = 0; e < per; e++) s += qa[e] * ks[t][lane * per + e];
                s = simd_sum(s) * a.scale;
                if (pass == 0u) { const float nm = max(mx, s); sum = sum * exp(mx - nm) + exp(s - nm); mx = nm; }
                else { const float p = v41_bf16r(exp(s - mx)); for (uint e = 0; e < per; e++) acc[e] += p * ks[t][lane * per + e]; }
            }
        }
    }
    const float den = sum + exp(sink[h] - mx);
    for (uint e = 0; e < per; e++) o[((ulong)r * a.n_head + h) * hd + lane * per + e] = v41_bf16r(acc[e] / den);
    if (lane == 0u) lse[(ulong)r * a.n_head + h] = mx + log(den);
}
// 反向: g_q += scale·g_s·k; 块内键 g_k += scale·g_s·q̃ + p·g_o(原子加); 历史键是常量
kernel void kernel_v41_draft_attn_bwd(constant v41_dk_args &a [[buffer(0)]], device float *gq [[buffer(1)]], device atomic_float *gblk [[buffer(2)]],
                                      device const float *go [[buffer(3)]], device const float *o [[buffer(4)]], device const float *q [[buffer(5)]],
                                      device const float *lse [[buffer(6)]], device const float *hist [[buffer(7)]], device const float *blk [[buffer(8)]],
                                      device const int *bpos [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                                      uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ks[8][512];
    const uint r = tg.x, h = tg.y * 8u + sgitg, hd = a.hd, per = hd / 32u, b = r / a.B;
    const uint ib = (uint)bpos[b], nh = ib < a.window ? ib : a.window, nkeys = nh + a.B;
    float qa[16], ga[16], gacc[16];
    float D = 0.0f;
    for (uint e = 0; e < per; e++) {
        const ulong base = ((ulong)r * a.n_head + h) * hd + lane * per + e;
        qa[e] = v41_bf16r(q[base]); ga[e] = go[base]; gacc[e] = 0.0f; D += go[base] * o[base];
    }
    D = simd_sum(D);
    const float lh = lse[(ulong)r * a.n_head + h];
    for (uint base = 0; base < nkeys; base += 8u) {
        const uint nt = (nkeys - base) < 8u ? (nkeys - base) : 8u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < nt) {
            device const float *krow = v41_dk_key_row(hist, a.hbase, blk, b, ib, nh, a.B, base + sgitg, hd);
            for (uint d = lane; d < hd; d += 32u) ks[sgitg][d] = v41_bf16r(krow[d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < nt; t++) {
            const uint kk = base + t;
            float s = 0.0f, dp = 0.0f;
            for (uint e = 0; e < per; e++) { s += qa[e] * ks[t][lane * per + e]; dp += ga[e] * ks[t][lane * per + e]; }
            s = simd_sum(s); dp = simd_sum(dp);
            const float p = exp(s * a.scale - lh), c = a.scale * p * (dp - D);
            for (uint e = 0; e < per; e++) gacc[e] += c * ks[t][lane * per + e];
            if (kk >= nh) {
                device atomic_float *gk = gblk + ((ulong)b * a.B + (kk - nh)) * hd + lane * per;
                for (uint e = 0; e < per; e++) atomic_fetch_add_explicit(gk + e, c * qa[e] + p * ga[e], memory_order_relaxed);
            }
        }
    }
    for (uint e = 0; e < per; e++) gq[((ulong)r * a.n_head + h) * hd + lane * per + e] = gacc[e];
}
// 陪审团: out[m][4] = {Σmin(p,q), p(argmax q), argmax 同, 0}; p = 教师(lt) q = 学生(ls), 温度 T
kernel void kernel_v41_draft_jury(device float *out [[buffer(0)]], device const float *ls [[buffer(1)]], device const float *lt [[buffer(2)]],
                                  constant uint &V [[buffer(3)]], constant float &T [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sf[1024]; threadgroup int si[1024];
    device const float *s = ls + (ulong)row * V, *t = lt + (ulong)row * V;
    float vs = -INFINITY, vt = -INFINITY; int is = -1, it = -1;
    for (uint i = tid; i < V; i += 1024u) {
        if (s[i] > vs) { vs = s[i]; is = (int)i; }
        if (t[i] > vt) { vt = t[i]; it = (int)i; }
    }
    float ms, mt; int as_, at;
    v41_tg_argmax(vs, is, sf, si, tid, 1024u, ms, as_);
    v41_tg_argmax(vt, it, sf, si, tid, 1024u, mt, at);
    float zs = 0.0f, zt = 0.0f;
    for (uint i = tid; i < V; i += 1024u) {
        zs += v41_finite(s[i]) ? precise::exp((s[i] - ms) / T) : 0.0f;
        zt += v41_finite(t[i]) ? precise::exp((t[i] - mt) / T) : 0.0f;
    }
    zs = v41_tg_sum(zs, sf, tid, 1024u); zt = v41_tg_sum(zt, sf, tid, 1024u);
    float smin = 0.0f;
    for (uint i = tid; i < V; i += 1024u) {
        const float p = v41_finite(t[i]) ? precise::exp((t[i] - mt) / T) / zt : 0.0f, qq = v41_finite(s[i]) ? precise::exp((s[i] - ms) / T) / zs : 0.0f;
        smin += min(p, qq);
    }
    smin = v41_tg_sum(smin, sf, tid, 1024u);
    if (tid == 0u) {
        out[(ulong)row * 4u] = smin;
        out[(ulong)row * 4u + 1u] = as_ >= 0 ? precise::exp((t[as_] - mt) / T) / zt : 0.0f;
        out[(ulong)row * 4u + 2u] = as_ == at ? 1.0f : 0.0f;
        out[(ulong)row * 4u + 3u] = 0.0f;
    }
}
// 总变差损失(训练目标 = 1 − 期望接受率): 学生按温度 T 做 softmax, 教师 top-K + 余量; 梯度 ∂TV/∂z_i = (1/T)·q_i·½·(s_i − Σ_j q_j s_j)
struct v41_tv_args { uint V, K, pad0, pad1; float T, scale, pad2, pad3; };
kernel void kernel_v41_draft_tv(constant v41_tv_args &a [[buffer(0)]], device float *g [[buffer(1)]], device float *loss [[buffer(2)]],
                                device const float *z [[buffer(3)]], device const int *tid_ [[buffer(4)]], device const float *tp [[buffer(5)]],
                                device const float *trest [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint mask[4096];
    threadgroup float sf[1024];
    const uint V = a.V, K = a.K; const float T = a.T;
    device const float *zr = z + (ulong)row * V; device float *gr = g + (ulong)row * V;
    device const int *ti = tid_ + (ulong)row * K; device const float *tq = tp + (ulong)row * K;
    for (uint w = tid; w < 4096u; w += 1024u) atomic_store_explicit(&mask[w], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < K; k += 1024u) { const int id = ti[k]; if (id >= 0 && (uint)id < V) atomic_fetch_or_explicit(&mask[(uint)id >> 5], 1u << ((uint)id & 31u), memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint j = tid; j < V; j += 1024u) mx = max(mx, zr[j]);
    mx = v41_tg_max(mx, sf, tid, 1024u);
    float Z = 0.0f;
    for (uint j = tid; j < V; j += 1024u) Z += v41_finite(zr[j]) ? precise::exp((zr[j] - mx) / T) : 0.0f;
    Z = v41_tg_sum(Z, sf, tid, 1024u);
    float qoff = 0.0f;
    for (uint j = tid; j < V; j += 1024u)
        if (!((atomic_load_explicit(&mask[j >> 5], memory_order_relaxed) >> (j & 31u)) & 1u)) qoff += v41_finite(zr[j]) ? precise::exp((zr[j] - mx) / T) / Z : 0.0f;
    qoff = v41_tg_sum(qoff, sf, tid, 1024u);
    float tvon = 0.0f, qson = 0.0f;
    for (uint k = tid; k < K; k += 1024u) {
        const int id = ti[k];
        if (id < 0 || (uint)id >= V) continue;
        const float q = v41_finite(zr[id]) ? precise::exp((zr[id] - mx) / T) / Z : 0.0f, d = q - tq[k];
        tvon += abs(d); qson += d > 0.0f ? q : (d < 0.0f ? -q : 0.0f);
    }
    tvon = v41_tg_sum(tvon, sf, tid, 1024u); qson = v41_tg_sum(qson, sf, tid, 1024u);
    const float A = qoff + qson, c = a.scale / T;
    for (uint j = tid; j < V; j += 1024u) {
        if ((atomic_load_explicit(&mask[j >> 5], memory_order_relaxed) >> (j & 31u)) & 1u) continue;
        const float q = v41_finite(zr[j]) ? precise::exp((zr[j] - mx) / T) / Z : 0.0f;
        gr[j] = c * q * 0.5f * (1.0f - A);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint k = tid; k < K; k += 1024u) {
        const int id = ti[k];
        if (id < 0 || (uint)id >= V) continue;
        const float q = v41_finite(zr[id]) ? precise::exp((zr[id] - mx) / T) / Z : 0.0f, d = q - tq[k], s = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        gr[id] = c * q * 0.5f * (s - A);
    }
    if (tid == 0u) loss[row] = 0.5f * (tvon + qoff + trest[row]);
}
// 设备表取行 / 梯度按行散加(同一 token 一批里出现多次 ⇒ 原子加)
kernel void kernel_v41_rows_dev(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *tab [[buffer(2)]],
                                device const int *ids [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint ids_off = a.n0, R = a.n1, Vm = a.n2, d = gid.x, r = gid.y;
    if (d >= R) return;
    const int id = ids[ids_off + r];
    out[(ulong)r * R + d] = (id >= 0 && (uint)id < Vm) ? tab[(ulong)id * R + d] : 0.0f;
}
kernel void kernel_v41_rows_scatter(constant v41_n_args &a [[buffer(0)]], device atomic_float *gtab [[buffer(1)]], device const float *g [[buffer(2)]],
                                    device const int *ids [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint R = a.n1, Vm = a.n2, d = gid.x, r = gid.y;
    if (d >= R) return;
    const int id = ids[r];
    if (id >= 0 && (uint)id < Vm) atomic_fetch_add_explicit(&gtab[(ulong)id * R + d], g[(ulong)r * R + d], memory_order_relaxed);
}
