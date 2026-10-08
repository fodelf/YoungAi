// v41_bwd.metal — 后训练反传原语(Metal, 2026-10-08), 对应 cuda_bwd_dense / cuda_bwd_attn / cuda_bwd_hc / cuda_bwd_moe / cuda_bwd_comp / cuda_bwd_vq。
// 口径: 激活与梯度 f32; 前向的 bf16 舍入当直通。转置乘走 v41_dense.metal 的 wgemm(wnn=1), 这里只有逐元素/归约类的核与 VQ 专家反向的直读两核。
// 跨 threadgroup 的累加(键梯度 / 表梯度)用 device 上的 atomic float(MSL 3.0+)。必须排在 v41_common.metal / v41_attn.metal / v41_vq.metal 之后。

// ---- 上下文蒸馏: 教师 top-K + 余量桶的前向 KL 与对学生 logits 的梯度(一 threadgroup 一行 1024 线程; 位图标榜上)
struct v41_kl_args { uint row0, V, K, has_w; float scale; float pad0, pad1, pad2; };
kernel void kernel_v41_bwd_kl_topk(constant v41_kl_args &a [[buffer(0)]], device float *g [[buffer(1)]], device float *loss [[buffer(2)]],
                                   device const float *z [[buffer(3)]], device const int *tid_ [[buffer(4)]], device const float *tp [[buffer(5)]],
                                   device const float *trest [[buffer(6)]], device const float *w [[buffer(7)]],
                                   uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[1024], ps_k[128], s_off;
    threadgroup atomic_uint mask[4096];
    const uint V = a.V, K = a.K;
    const float wi = a.has_w ? w[i] : 1.0f, scale = a.scale * wi;
    device const float *zr = z + (ulong)(a.row0 + i) * V;
    device float *gr = g + (ulong)i * V;
    device const int *ids = tid_ + (ulong)i * K; device const float *pt = tp + (ulong)i * K;
    for (uint q = tid; q < 4096u; q += 1024u) atomic_store_explicit(&mask[q], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < K; k += 1024u) { const int id = ids[k]; if (id >= 0 && (uint)id < V) atomic_fetch_or_explicit(&mask[(uint)id >> 5], 1u << ((uint)id & 31u), memory_order_relaxed); }
    float mx = -INFINITY;
    for (uint j = tid; j < V; j += 1024u) mx = max(mx, zr[j]);
    mx = v41_tg_max(mx, sh, tid, 1024u);
    float s = 0.0f, sr = 0.0f;
    for (uint j = tid; j < V; j += 1024u) {
        const float ev = exp(zr[j] - mx);
        s += ev;
        if (!((atomic_load_explicit(&mask[j >> 5], memory_order_relaxed) >> (j & 31u)) & 1u)) sr += ev;
    }
    s = v41_tg_sum(s, sh, tid, 1024u); sr = v41_tg_sum(sr, sh, tid, 1024u);
    const float lse = mx + log(s), rs = sr / s;
    for (uint k = tid; k < K; k += 1024u) { const int id = ids[k]; ps_k[k] = (id >= 0 && (uint)id < V) ? exp(zr[id] - lse) : 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float l = 0.0f;
        for (uint k = 0; k < K; k++) if (pt[k] > 0.0f) l += pt[k] * (log(pt[k]) - log(max(ps_k[k], 1e-30f)));
        const float rt = max(trest[i], 0.0f);
        if (rt > 1e-12f) l += rt * (log(rt) - log(max(rs, 1e-30f)));
        loss[i] = l * wi;
        s_off = rs > 1e-30f ? 1.0f - rt / rs : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float off = s_off;
    for (uint j = tid; j < V; j += 1024u) gr[j] = scale * exp(zr[j] - lse) * off;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint k = tid; k < K; k += 1024u) { const int id = ids[k]; if (id >= 0 && (uint)id < V) gr[id] = scale * (ps_k[k] - pt[k]); }
}
// 教师 top-K: 整行 lse, 再 K 轮"全块 argmax(同值取小下标) → 置 −inf"; 余量 = 榜外直接累加
kernel void kernel_v41_bwd_topk(constant v41_kl_args &a [[buffer(0)]], device int *tid_ [[buffer(1)]], device float *tp [[buffer(2)]],
                                device float *trest [[buffer(3)]], device float *z [[buffer(4)]],
                                uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[1024]; threadgroup int si[1024];
    const uint V = a.V, K = a.K;
    device float *zr = z + (ulong)(a.row0 + i) * V;
    float mx = -INFINITY;
    for (uint j = tid; j < V; j += 1024u) mx = max(mx, zr[j]);
    mx = v41_tg_max(mx, sh, tid, 1024u);
    float s = 0.0f;
    for (uint j = tid; j < V; j += 1024u) s += exp(zr[j] - mx);
    s = v41_tg_sum(s, sh, tid, 1024u);
    for (uint k = 0; k < K; k++) {
        float bv = -INFINITY; int bi = -1;
        for (uint j = tid; j < V; j += 1024u) { const float v = zr[j]; if (v > bv) { bv = v; bi = (int)j; } }
        float v0; int i0;
        v41_tg_argmax(bv, bi, sh, si, tid, 1024u, v0, i0);
        if (tid == 0u) {
            const bool none = i0 < 0 || v0 == -INFINITY;
            tid_[(ulong)i * K + k] = none ? -1 : i0;
            tp[(ulong)i * K + k] = none ? 0.0f : exp(v0 - mx) / s;
            if (!none) zr[i0] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    float sr = 0.0f;
    for (uint j = tid; j < V; j += 1024u) sr += exp(zr[j] - mx);
    sr = v41_tg_sum(sr, sh, tid, 1024u);
    if (tid == 0u) trest[i] = sr / s;
}
// RMSNorm 反向: gx = r·(w⊙gxn) − x·r³·Σ(w⊙gxn⊙x)/D
kernel void kernel_v41_bwd_rms_norm(constant v41_n_args &a [[buffer(0)]], device float *gx [[buffer(1)]], device const float *gxn [[buffer(2)]],
                                    device const float *x [[buffer(3)]], device const float *w [[buffer(4)]], constant ulong &w_off [[buffer(5)]],
                                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[512];
    const uint D = a.n0, accf = a.n1; const float eps = a.f0;
    device const float *wr = (device const float *)((device const uchar *)w + w_off);
    device const float *xr = x + (ulong)r * D, *gr = gxn + (ulong)r * D;
    float ss = 0.0f, dot = 0.0f;
    for (uint d = tid; d < D; d += 512u) { const float xv = xr[d]; ss += xv * xv; dot += wr[d] * gr[d] * xv; }
    ss = v41_tg_sum(ss, sh, tid, 512u); dot = v41_tg_sum(dot, sh, tid, 512u);
    const float inv = rsqrt(ss / float(D) + eps), c = inv * inv * inv * dot / float(D);
    device float *o = gx + (ulong)r * D;
    for (uint d = tid; d < D; d += 512u) { const float v = inv * wr[d] * gr[d] - xr[d] * c; o[d] = accf ? o[d] + v : v; }
}
// hc_pre 反向: ghc[c][d] += pre[c]·gx[d]; gpre[c] = Σ_d gx[d]·hc[c][d]
kernel void kernel_v41_bwd_hc_pre(constant v41_n_args &a [[buffer(0)]], device float *ghc [[buffer(1)]], device float *gpre [[buffer(2)]],
                                  device const float *gx [[buffer(3)]], device const float *hc [[buffer(4)]], device const float *pre [[buffer(5)]],
                                  uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[512];
    const uint E = a.n0, HC = a.n1;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint d = tid; d < E; d += 512u) {
        const float g = gx[(ulong)n * E + d];
        for (uint c = 0; c < HC; c++) { const ulong o = ((ulong)n * HC + c) * E + d; acc[c] += g * hc[o]; ghc[o] += pre[n * HC + c] * g; }
    }
    for (uint c = 0; c < HC; c++) { const float v = v41_tg_sum(acc[c], sh, tid, 512u); if (tid == 0u) gpre[n * HC + c] = v; }
}
// hc_post 反向(可选输出由 has_* 标记): gy = Σ_k post[k]·gout[k]; gres[j] += Σ_k comb[j][k]·gout[k]; gpost[k] = Σ_d gout[k]·y; gcomb[j][k] = Σ_d gout[k]·res[j]
kernel void kernel_v41_bwd_hc_post(constant v41_n_args &a [[buffer(0)]], device float *gy [[buffer(1)]], device float *gres [[buffer(2)]],
                                   device float *gpost [[buffer(3)]], device float *gcomb [[buffer(4)]], device const float *gout [[buffer(5)]],
                                   device const float *y [[buffer(6)]], device const float *res [[buffer(7)]], device const float *post [[buffer(8)]],
                                   device const float *comb [[buffer(9)]], uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[512];
    const uint E = a.n0, HC = a.n1, has_gres = a.n2 & 1u, has_gpost = (a.n2 >> 1) & 1u, has_gcomb = (a.n2 >> 2) & 1u;
    device const float *cb = comb + (ulong)n * HC * HC, *po = post + (ulong)n * HC;
    float ap[4] = { 0, 0, 0, 0 }, ac[16];
    for (uint q = 0; q < 16u; q++) ac[q] = 0.0f;
    for (uint d = tid; d < E; d += 512u) {
        float go[4], rv[4];
        for (uint k = 0; k < HC; k++) go[k] = gout[((ulong)n * HC + k) * E + d];
        for (uint j = 0; j < HC; j++) rv[j] = res[((ulong)n * HC + j) * E + d];
        const float yv = y[(ulong)n * E + d];
        float s = 0.0f;
        for (uint k = 0; k < HC; k++) { s += po[k] * go[k]; ap[k] += go[k] * yv; }
        gy[(ulong)n * E + d] = s;
        for (uint j = 0; j < HC; j++) {
            float r = 0.0f;
            for (uint k = 0; k < HC; k++) { r += cb[j * HC + k] * go[k]; ac[j * 4u + k] += go[k] * rv[j]; }
            if (has_gres) gres[((ulong)n * HC + j) * E + d] += r;
        }
    }
    for (uint k = 0; k < HC; k++) { const float v = v41_tg_sum(ap[k], sh, tid, 512u); if (tid == 0u && has_gpost) gpost[n * HC + k] = v; }
    for (uint j = 0; j < HC; j++) for (uint k = 0; k < HC; k++) { const float v = v41_tg_sum(ac[j * 4u + k], sh, tid, 512u); if (tid == 0u && has_gcomb) gcomb[(ulong)n * HC * HC + j * HC + k] = v; }
}
// Σg² 的逐 threadgroup 部分和(主机再按序求和 → 确定性)
kernel void kernel_v41_bwd_sumsq(device float *part [[buffer(0)]], device const float *g [[buffer(1)]], constant ulong &n [[buffer(2)]],
                                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
    threadgroup float sh[512];
    float s = 0.0f;
    for (ulong i = (ulong)tg * 512u + tid; i < n; i += (ulong)ntg * 512u) s += g[i] * g[i];
    const float r = v41_tg_sum(s, sh, tid, 512u);
    if (tid == 0u) part[tg] = r;
}
struct v41_adam_args { ulong n; float lr, b1, b2, eps, gs, c1, c2, pad; };
kernel void kernel_v41_bwd_adam(constant v41_adam_args &a [[buffer(0)]], device float *p [[buffer(1)]], device float *g [[buffer(2)]],
                                device float *m [[buffer(3)]], device float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= a.n) return;
    const float gi = g[i] * a.gs;
    const float mi = a.b1 * m[i] + (1.0f - a.b1) * gi, vi = a.b2 * v[i] + (1.0f - a.b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    p[i] -= a.lr * (mi / a.c1) / (sqrt(vi / a.c2) + a.eps);
    g[i] = 0.0f;
}
kernel void kernel_v41_pack_bf16(device ushort *o [[buffer(0)]], device const float *x [[buffer(1)]], constant ulong &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) o[i] = v41_f32_to_bf16(x[i]);
}
kernel void kernel_v41_unpack_bf16(device float *o [[buffer(0)]], device const ushort *x [[buffer(1)]], constant ulong &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) o[i] = v41_bf16_to_f32(x[i]);
}

// ---- 稀疏注意力反向(一题一块、pos0 = 0; 窗口缓冲第 window+i 行 = 位置 i): 一 threadgroup 一 (query, 8 头), 8 键一片进 threadgroup;
// 两遍: 先 lse(与前向同序的在线 max/sum), 再 g_s = p·(dp − D) → g_q += scale·g_s·k; 键梯度 scale·g_s·q + p·g_o 原子加进 gkv[位置] / gcomp[组]。
struct v41_battn_args { uint window, ng, topk, n_head, hd, has_comp, has_gcomp, pad; float scale; float pad1, pad2, pad3; };
kernel void kernel_v41_bwd_sparse_attn(constant v41_battn_args &a [[buffer(0)]], device float *gq [[buffer(1)]], device atomic_float *gkv [[buffer(2)]],
                                       device atomic_float *gcomp [[buffer(3)]], device const float *go [[buffer(4)]], device const float *o [[buffer(5)]],
                                       device const float *q [[buffer(6)]], device const float *kvw [[buffer(7)]], device const uchar *kvc [[buffer(8)]],
                                       device const int *idx [[buffer(9)]], device const float *sink [[buffer(10)]],
                                       uint2 tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ks[8][512], ksc[8][32];
    threadgroup int kok[8], kpos[8];
    const uint i = tg.x, h = tg.y * 8u + sgitg, hd = a.hd, per = hd / 32u;
    const uint lo = i + 1u > a.window ? i + 1u - a.window : 0u, nwin = i - lo + 1u, topk = a.has_comp ? a.topk : 0u, nkeys = nwin + topk;
    float qa[16], ga[16], gacc[16];
    float D = 0.0f;
    for (uint e = 0; e < per; e++) {
        const ulong b = ((ulong)i * a.n_head + h) * hd + lane * per + e;
        qa[e] = q[b]; ga[e] = go[b]; gacc[e] = 0.0f; D += go[b] * o[b];
    }
    D = simd_sum(D);
    float mx = -1e30f, sum = 0.0f;
    for (uint pass = 0; pass < 2u; pass++) {
        const float lse = pass ? mx + log(sum + exp(sink[h] - mx)) : 0.0f;
        for (uint base = 0; base < nkeys; base += 8u) {
            const uint nt = (nkeys - base) < 8u ? (nkeys - base) : 8u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgitg < nt) {
                const uint t = sgitg, kk = base + t;
                bool have_w = false, have_c = false; ulong wrow = 0; device const uchar *cpk = kvc; int g = -1;
                if (kk < nwin) { have_w = true; wrow = (ulong)(a.window + lo + kk) * hd; }
                else { g = idx[(ulong)i * topk + (kk - nwin)]; if (g >= 0 && (uint)g < a.ng) { have_c = true; cpk = kvc + (ulong)g * 288u; } }
                if (lane == 0u) { kok[t] = (have_w || have_c) ? 1 : 0; kpos[t] = have_w ? (int)(lo + kk) : (have_c ? -2 - g : -1); }
                if (have_c) ksc[t][lane] = v41_e4m3_to_f32(cpk[256u + lane]);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint d = lane; d < hd; d += 32u) ks[t][d] = have_w ? kvw[wrow + d] : (have_c ? v41_ckv_get_s(cpk, d, ksc[t]) : 0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (pass == 0u) {
                float s[8]; float tm = -1e30f;
                for (uint t = 0; t < nt; t++) {
                    float d = 0.0f;
                    for (uint e = 0; e < per; e++) d += qa[e] * ks[t][lane * per + e];
                    d = simd_sum(d);
                    s[t] = kok[t] ? d * a.scale : -1e30f; tm = max(tm, s[t]);
                }
                const float nm = max(mx, tm), rs = exp(mx - nm);
                float ps = 0.0f;
                for (uint t = 0; t < nt; t++) ps += exp(s[t] - nm);
                sum = sum * rs + ps; mx = nm;
            } else {
                for (uint t = 0; t < nt; t++) {
                    if (!kok[t]) continue;
                    float s = 0.0f, dp = 0.0f;
                    for (uint e = 0; e < per; e++) { s += qa[e] * ks[t][lane * per + e]; dp += ga[e] * ks[t][lane * per + e]; }
                    s = simd_sum(s); dp = simd_sum(dp);
                    const float p = exp(s * a.scale - lse), c = a.scale * p * (dp - D);
                    for (uint e = 0; e < per; e++) gacc[e] += c * ks[t][lane * per + e];
                    const int kp = kpos[t];
                    if (kp >= 0) { for (uint e = 0; e < per; e++) atomic_fetch_add_explicit(&gkv[(ulong)kp * hd + lane * per + e], c * qa[e] + p * ga[e], memory_order_relaxed); }
                    else if (kp <= -2 && a.has_gcomp) { const uint g = (uint)(-2 - kp); for (uint e = 0; e < per; e++) atomic_fetch_add_explicit(&gcomp[(ulong)g * hd + lane * per + e], c * qa[e] + p * ga[e], memory_order_relaxed); }
                }
            }
        }
    }
    for (uint e = 0; e < per; e++) gq[((ulong)i * a.n_head + h) * hd + lane * per + e] = gacc[e];
}

// ---- mHC 混合系数反向: ① 每 token 一线程重算 sinkhorn 并倒推 g_C0 → g_mix[24]; ② g_h += inv·Wᵀ·g_mix − inv³/dim·<g_mix, W·h>·h
struct v41_bhc_args { uint n_tok, hc, iters, mh, dim, has_gpre, has_gpost, has_gcomb; float eps, norm_eps, pad0, pad1; };
inline void v41_sinkhorn_back(thread float *gC0, const thread float *C0, const thread float *gout, uint hc, uint iters, float eps) {
    float P[64][16], sm[16];
    for (uint j = 0; j < hc; j++) {
        float mx = -INFINITY; for (uint k = 0; k < hc; k++) mx = max(mx, C0[j * hc + k]);
        float s = 0.0f; for (uint k = 0; k < hc; k++) s += exp(C0[j * hc + k] - mx);
        for (uint k = 0; k < hc; k++) { sm[j * hc + k] = exp(C0[j * hc + k] - mx) / s; P[0][j * hc + k] = sm[j * hc + k] + eps; }
    }
    uint np = 1u;
    for (uint step = 1; step < 2u * iters; step++) {
        if (step & 1u) { for (uint k = 0; k < hc; k++) { float s = 0.0f; for (uint j = 0; j < hc; j++) s += P[np - 1][j * hc + k]; for (uint j = 0; j < hc; j++) P[np][j * hc + k] = P[np - 1][j * hc + k] / (s + eps); } }
        else { for (uint j = 0; j < hc; j++) { float s = 0.0f; for (uint k = 0; k < hc; k++) s += P[np - 1][j * hc + k]; for (uint k = 0; k < hc; k++) P[np][j * hc + k] = P[np - 1][j * hc + k] / (s + eps); } }
        np++;
    }
    float g[16], ga[16];
    for (uint q = 0; q < hc * hc; q++) g[q] = gout[q];
    for (uint step = 2u * iters - 1u; step >= 1u; step--) {
        if (step & 1u) {
            for (uint k = 0; k < hc; k++) {
                float s = 0.0f, t = 0.0f;
                for (uint j = 0; j < hc; j++) { s += P[step - 1][j * hc + k]; t += g[j * hc + k] * P[step - 1][j * hc + k]; }
                const float den = s + eps;
                for (uint j = 0; j < hc; j++) ga[j * hc + k] = g[j * hc + k] / den - t / (den * den);
            }
        } else {
            for (uint j = 0; j < hc; j++) {
                float s = 0.0f, t = 0.0f;
                for (uint k = 0; k < hc; k++) { s += P[step - 1][j * hc + k]; t += g[j * hc + k] * P[step - 1][j * hc + k]; }
                const float den = s + eps;
                for (uint k = 0; k < hc; k++) ga[j * hc + k] = g[j * hc + k] / den - t / (den * den);
            }
        }
        for (uint q = 0; q < hc * hc; q++) g[q] = ga[q];
    }
    for (uint j = 0; j < hc; j++) {
        float t = 0.0f; for (uint k = 0; k < hc; k++) t += g[j * hc + k] * sm[j * hc + k];
        for (uint k = 0; k < hc; k++) gC0[j * hc + k] = sm[j * hc + k] * (g[j * hc + k] - t);
    }
}
kernel void kernel_v41_bwd_hc_split(constant v41_bhc_args &a [[buffer(0)]], device float *gmix [[buffer(1)]], device const float *mix [[buffer(2)]],
                                    device const float *gpre [[buffer(3)]], device const float *gpost [[buffer(4)]], device const float *gcomb [[buffer(5)]],
                                    device const float *scale [[buffer(6)]], device const float *base [[buffer(7)]], uint n [[thread_position_in_grid]]) {
    if (n >= a.n_tok) return;
    const uint hc = a.hc, mh = a.mh;
    device const float *m = mix + (ulong)n * mh; device float *gm = gmix + (ulong)n * mh;
    for (uint c = 0; c < hc; c++) {
        const float sp = 1.0f / (1.0f + exp(-(m[c] * scale[0] + base[c])));
        gm[c] = (a.has_gpre ? gpre[n * hc + c] : 0.0f) * sp * (1.0f - sp) * scale[0];
        const float so = 1.0f / (1.0f + exp(-(m[hc + c] * scale[1] + base[hc + c])));
        gm[hc + c] = (a.has_gpost ? gpost[n * hc + c] : 0.0f) * 2.0f * so * (1.0f - so) * scale[1];
    }
    float C0[16], gC0[16], go[16];
    for (uint q = 0; q < hc * hc; q++) { C0[q] = m[2u * hc + q] * scale[2] + base[2u * hc + q]; go[q] = a.has_gcomb ? gcomb[(ulong)n * hc * hc + q] : 0.0f; }
    v41_sinkhorn_back(gC0, C0, go, hc, a.iters, a.eps);
    for (uint q = 0; q < hc * hc; q++) gm[2u * hc + q] = gC0[q] * scale[2];
}
kernel void kernel_v41_bwd_hc_mix(constant v41_bhc_args &a [[buffer(0)]], device float *gh [[buffer(1)]], device const float *gmix [[buffer(2)]],
                                  device const float *mix [[buffer(3)]], device const float *W [[buffer(4)]], device const float *h [[buffer(5)]],
                                  uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[512], gm[32];
    const uint dim = a.dim, mh = a.mh;
    device const float *hr = h + (ulong)n * dim;
    float ss = 0.0f;
    for (uint i = tid; i < dim; i += 512u) ss += hr[i] * hr[i];
    ss = v41_tg_sum(ss, sh, tid, 512u);
    const float inv = rsqrt(ss / float(dim) + a.norm_eps);
    if (tid < mh) gm[tid] = gmix[(ulong)n * mh + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint r = 0; r < mh; r++) dot += gm[r] * mix[(ulong)n * mh + r];
    dot /= inv;
    const float c = inv * inv * inv * dot / float(dim);
    for (uint i = tid; i < dim; i += 512u) {
        float s = 0.0f;
        for (uint r = 0; r < mh; r++) s += W[(ulong)r * dim + i] * gm[r];
        gh[(ulong)n * dim + i] += inv * s - c * hr[i];
    }
}
// ---- MoE 一族: SwiGLU 反向 / 路由反向 / 冻结选择前向 / 配对收放 / 行点积
kernel void kernel_v41_bwd_swiglu(device float *gg [[buffer(0)]], device float *gu [[buffer(1)]], device const float *gh [[buffer(2)]],
                                  device const float *g [[buffer(3)]], device const float *u [[buffer(4)]], constant ulong &n [[buffer(5)]],
                                  constant float &L [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= n) return;
    float gv = g[i], uv = u[i];
    const bool gpass = !(L > 0.0f) || gv < L, upass = !(L > 0.0f) || (uv > -L && uv < L);
    if (L > 0.0f) { uv = min(max(uv, -L), L); gv = min(gv, L); }
    const float sg = 1.0f / (1.0f + exp(-gv)), si = gv * sg, ghv = gh[i];
    gg[i] = gpass ? ghv * uv * sg * (1.0f + gv * (1.0f - sg)) : 0.0f;
    gu[i] = upass ? ghv * si : 0.0f;
}
// a = swiglu(hg, hu)(前向重算, f32 存)
kernel void kernel_v41_bwd_swiglu_fwd(device float *o [[buffer(0)]], device const float *hg [[buffer(1)]], device const float *hu [[buffer(2)]],
                                      constant ulong &n [[buffer(3)]], constant float &L [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= n) return;
    float gv = hg[i], uv = hu[i];
    if (L > 0.0f) { uv = min(max(uv, -L), L); gv = min(gv, L); }
    o[i] = v41_bf16r((gv / (1.0f + exp(-gv))) * uv);
}
struct v41_brt_args { uint NE, K, pad0, pad1; float rs; float pad2, pad3, pad4; };
kernel void kernel_v41_bwd_router(constant v41_brt_args &a [[buffer(0)]], device float *gz [[buffer(1)]], device const float *gw [[buffer(2)]],
                                  device const int *sel [[buffer(3)]], device const float *z [[buffer(4)]],
                                  uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint NE = a.NE, K = a.K;
    device float *gr = gz + (ulong)t * NE;
    for (uint e = tid; e < NE; e += 128u) gr[e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid != 0u) return;
    float pr[16], S = 0.0f;
    for (uint k = 0; k < K; k++) {
        const int e = sel[(ulong)t * K + k];
        const float zz = e >= 0 ? z[(ulong)t * NE + e] : 0.0f;
        const float sp = zz > 20.0f ? zz : log(1.0f + exp(zz));
        pr[k] = e >= 0 ? sqrt(sp) : 0.0f; S += pr[k];
    }
    S += 1e-20f;
    float tw = 0.0f;
    for (uint k = 0; k < K; k++) tw += gw[(ulong)t * K + k] * pr[k] / S;
    for (uint k = 0; k < K; k++) {
        const int e = sel[(ulong)t * K + k];
        if (e < 0 || pr[k] <= 0.0f) continue;
        const float gpr = a.rs / S * (gw[(ulong)t * K + k] - tw), zz = z[(ulong)t * NE + e];
        const float dsp = zz > 20.0f ? 1.0f : 1.0f / (1.0f + exp(-zz));
        gr[e] += gpr * dsp / (2.0f * pr[k]);
    }
}
kernel void kernel_v41_bwd_router_fixed(constant v41_brt_args &a [[buffer(0)]], device float *wts [[buffer(1)]], device const int *sel [[buffer(2)]],
                                        device const float *logits [[buffer(3)]], constant uint &n_tok [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if (t >= n_tok) return;
    float pr[16], wsum = 0.0f;
    for (uint r = 0; r < a.K; r++) {
        const int e = sel[(ulong)t * a.K + r];
        float p = 0.0f;
        if (e >= 0 && (uint)e < a.NE) { const float z = logits[(ulong)t * a.NE + e]; p = sqrt(z > 20.0f ? z : log(1.0f + exp(z))); }
        pr[r] = p; wsum += p;
    }
    for (uint r = 0; r < a.K; r++) wts[(ulong)t * a.K + r] = pr[r] / (wsum + 1e-20f) * a.rs;
}
// 按配对表收行: dst[r] = src[pair[r]/K]·(rw ? rw[pair[r]] : 1); 行点积 gw[pair[r]] = <gy[token], O[r]>; 放梯度(确定序): gx[t] += Σ_k og[inv[t·K+k]]
kernel void kernel_v41_bwd_gather(constant v41_n_args &a [[buffer(0)]], device float *dst [[buffer(1)]], device const float *src [[buffer(2)]],
                                  device const int *pair [[buffer(3)]], device const float *rw [[buffer(4)]],
                                  uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint K = a.n0, D = a.n1, has_rw = a.n2;
    const int p = pair[r];
    const float s = has_rw ? rw[p] : 1.0f;
    device const float *sr = src + (ulong)(p / (int)K) * D;
    for (uint d = tid; d < D; d += 256u) dst[(ulong)r * D + d] = sr[d] * s;
}
kernel void kernel_v41_bwd_rowdot(constant v41_n_args &a [[buffer(0)]], device float *gw [[buffer(1)]], device const float *gy [[buffer(2)]],
                                  device const float *O [[buffer(3)]], device const int *pair [[buffer(4)]],
                                  uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    const uint K = a.n0, D = a.n1;
    const int p = pair[r];
    device const float *g = gy + (ulong)(p / (int)K) * D, *o = O + (ulong)r * D;
    float s = 0.0f;
    for (uint d = tid; d < D; d += 256u) s += g[d] * o[d];
    s = v41_tg_sum(s, sh, tid, 256u);
    if (tid == 0u) gw[p] = s;
}
kernel void kernel_v41_bwd_scatter(constant v41_n_args &a [[buffer(0)]], device float *gx [[buffer(1)]], device const float *og [[buffer(2)]],
                                   device const int *inv [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint K = a.n0, D = a.n1, d = gid.x, t = gid.y;
    if (d >= D) return;
    float s = 0.0f;
    for (uint k = 0; k < K; k++) { const int i = inv[(ulong)t * K + k]; if (i >= 0) s += og[(ulong)i * D + d]; }
    gx[(ulong)t * D + d] += s;
}
// VQ 直读: 行点积 out[pos][r] = gain·Σ W[r]·x[pos](x 按排序位置 [nv][C]); 转置累加 out[pos][c] (+)= Σ_r g[pos][r]·gain_r·W[r][c]
struct v41_bvq_args { uint which, R, C, n_act, v3, has_gov, OUTd, accumulate; uint do_round, pad0, pad1, pad2; };
kernel void kernel_v41_vqb_rowdot(constant v41_bvq_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *x [[buffer(2)]],
                                  device const uchar *blob [[buffer(3)]], device const uint *meta [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                                  device const float *gov [[buffer(6)]], device atomic_int *bad [[buffer(7)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint k = tg.y, e = meta[k], p0 = meta[a.n_act + k], mcnt = meta[2u * a.n_act + k], r = tg.x * 8u + sgitg;
    if (r >= a.R) return;
    const v41_vq_mat m = v41_vq_open(blob + blob_off, e, a.which, a.R, a.C, a.v3);
    if (!m.ok) { if (lane == 0u) atomic_store_explicit(bad, 1, memory_order_relaxed); return; }
    const bool g2 = a.which == 2u && a.has_gov;
    v41_vq_rowdot_grp(m, r, x, a.C, 1u, (device const int *)meta, p0, mcnt, out, a.R, gov + (ulong)e * a.OUTd, g2 ? 1u : 0u, lane, 2u);
}
#define V41_VQB_MT 16u
#define V41_VQB_RT 64u
kernel void kernel_v41_vqb_tdot(constant v41_bvq_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *g [[buffer(2)]],
                                device const uchar *blob [[buffer(3)]], device const uint *meta [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                                device const float *gov [[buffer(6)]], device atomic_int *bad [[buffer(7)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float gs[V41_VQB_MT][V41_VQB_RT], gn[V41_VQB_RT];
    const uint k = tg.y, e = meta[k], p0 = meta[a.n_act + k], mcnt = meta[2u * a.n_act + k], j = tg.x * 128u + tid;
    const v41_vq_mat m = v41_vq_open(blob + blob_off, e, a.which, a.R, a.C, a.v3);
    if (!m.ok) { if (tid == 0u) atomic_store_explicit(bad, 1, memory_order_relaxed); return; }
    const bool g2 = a.which == 2u && a.has_gov, live = j < m.nidx_row;
    device const float *gv = gov + (ulong)e * a.OUTd;
    for (uint t0 = 0; t0 < mcnt; t0 += V41_VQB_MT) {
        const uint mt = mcnt - t0 < V41_VQB_MT ? mcnt - t0 : V41_VQB_MT;
        float acc[V41_VQB_MT][8];
        for (uint t = 0; t < V41_VQB_MT; t++) for (uint q = 0; q < 8u; q++) acc[t][q] = 0.0f;
        for (uint r0 = 0; r0 < a.R; r0 += V41_VQB_RT) {
            const uint nr = a.R - r0 < V41_VQB_RT ? a.R - r0 : V41_VQB_RT;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = tid; q < V41_VQB_MT * V41_VQB_RT; q += 128u) {
                const uint t = q / V41_VQB_RT, rr = q % V41_VQB_RT;
                gs[t][rr] = (t < mt && rr < nr) ? g[(ulong)(p0 + t0 + t) * a.R + r0 + rr] : 0.0f;
            }
            for (uint rr = tid; rr < V41_VQB_RT; rr += 128u) gn[rr] = rr < nr ? v41_vq_gain(m, r0 + rr, gv, g2 ? 1u : 0u) : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!live) continue;
            for (uint rr = 0; rr < nr; rr++) {
                float c[8];
                v41_vq_cw(m, v41_vq_code(m, r0 + rr, j), c);
                const float gsc = gn[rr];
                for (uint t = 0; t < V41_VQB_MT; t++) { const float w = gs[t][rr] * gsc; for (uint q = 0; q < 8u; q++) acc[t][q] += w * c[q]; }
            }
        }
        if (live) for (uint t = 0; t < mt; t++) {
            device float *o = out + (ulong)(p0 + t0 + t) * a.C + (ulong)j * 8u;
            for (uint q = 0; q < 8u; q++) o[q] = a.accumulate ? o[q] + acc[t][q] : acc[t][q];
        }
    }
}
// ---- 分界层以下: 压缩器池化反向 / engram 门反向
kernel void kernel_v41_bwd_compress_pool(constant v41_n_args &a [[buffer(0)]], device float *gkv [[buffer(1)]], device float *gsc [[buffer(2)]],
                                         device const float *gp [[buffer(3)]], device const float *kv [[buffer(4)]], device const float *sc [[buffer(5)]],
                                         uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint ratio = a.n0, dim = a.n1;
    for (uint d = tid; d < dim; d += 256u) {
        float mx = -INFINITY;
        for (uint t = 0; t < ratio; t++) mx = max(mx, sc[((ulong)g * ratio + t) * dim + d]);
        float den = 0.0f, acc = 0.0f;
        for (uint t = 0; t < ratio; t++) { const float e = exp(sc[((ulong)g * ratio + t) * dim + d] - mx); den += e; acc += e * kv[((ulong)g * ratio + t) * dim + d]; }
        const float P = acc / den, gP = gp[(ulong)g * dim + d];
        for (uint t = 0; t < ratio; t++) {
            const ulong o = ((ulong)g * ratio + t) * dim + d;
            const float w = exp(sc[o] - mx) / den;
            gkv[o] = w * gP; gsc[o] = w * (kv[o] - P) * gP;
        }
    }
}
kernel void kernel_v41_bwd_engram_gate(constant v41_n_args &a [[buffer(0)]], device float *g [[buffer(1)]], device const float *hc [[buffer(2)]],
                                       device const float *kv [[buffer(3)]], device const float *qw [[buffer(4)]], device const float *kw [[buffer(5)]],
                                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    const uint E = a.n0, n_hc = a.n1, c = tg.x, t = tg.y; const float eps = a.f0;
    device const float *h = hc + ((ulong)t * n_hc + c) * E;
    device float *gr = g + ((ulong)t * n_hc + c) * E;
    device const float *key = kv + (ulong)t * (n_hc + 1u) * E + (ulong)c * E, *val = kv + (ulong)t * (n_hc + 1u) * E + (ulong)n_hc * E;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint d = tid; d < E; d += 256u) {
        const float hv = h[d], kv_ = key[d], w = qw[c * E + d] * kw[c * E + d];
        s0 += hv * hv; s1 += kv_ * kv_; s2 += hv * w * kv_; s3 += gr[d] * val[d];
    }
    s0 = v41_tg_sum(s0, sh, tid, 256u); s1 = v41_tg_sum(s1, sh, tid, 256u); s2 = v41_tg_sum(s2, sh, tid, 256u); s3 = v41_tg_sum(s3, sh, tid, 256u);
    const float rh = rsqrt(s0 / float(E) + eps), rk = rsqrt(s1 / float(E) + eps), S = s2, ie = rsqrt(float(E));
    const float dot = S * rh * rk * ie, ad = abs(dot);
    const float z = copysign(sqrt(max(ad, 1e-6f)), dot), gate = 1.0f / (1.0f + exp(-z));
    const float gdot = ad > 1e-6f ? s3 * gate * (1.0f - gate) * 0.5f / sqrt(ad) : 0.0f;
    const float coef = gdot * rk * ie, hs = S * rh * rh * rh / float(E);
    for (uint d = tid; d < E; d += 256u) gr[d] += coef * (qw[c * E + d] * kw[c * E + d] * key[d] * rh - hs * h[d]);
}
