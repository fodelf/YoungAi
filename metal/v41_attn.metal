// v41_attn.metal — DeepSeek V4.1 稀疏注意力与 indexer 三件(Metal, 2026-10-08)。
// 对应 CUDA: cuda_v41_2(v41_sparse_attn_kernel 的按键分块版) / cuda_v41_indexer(打分 / 候选块 radix select / topk radix select)。
// 这里只有标量版: 一 threadgroup 8 个 simdgroup = 8 个头(CUDA 是 4 warp × 2 头, 键序与在线 softmax 的分块(8 键一块)一致)。
// posd 口径同 CUDA(ds4_gpu_v41.h "设备位置"): 非 NULL 时位置从 posd[0] 读, 主机的 ng/topk 当上限。必须排在 v41_common.metal 之后。

struct v41_attn_args { uint pos0, window, ng, topk, n_head, hd, full_block, ring, win_lo, has_posd, has_comp, pad; float scale; float pad1, pad2, pad3; };
// 窗口缓冲里绝对位置 a 住第几行(decode.md D1): 本批段 [window, window+n); 历史段 ring=1 环 / ring=0 线性
inline uint v41_win_row(int a, uint pos0, uint window, uint ring) {
    if (a >= (int)pos0) return window + (uint)(a - (int)pos0);
    return ring ? (uint)(a % (int)window) : (uint)(a - ((int)pos0 - (int)window));
}
inline float v41_ckv_get_s(device const uchar *row, uint d, threadgroup const float *scales) {
    const uint by = row[d >> 1];
    const uint nib = (d & 1u) ? (by >> 4) : (by & 0x0Fu);
    return v41_bf16r(v41_fp4_to_f32(nib) * scales[d >> 4]);
}
inline float v41_idxk_get(device const uchar *row, uint d) {
    const uint by = row[d >> 1];
    const uint nib = (d & 1u) ? (by >> 4) : (by & 0x0Fu);
    return v41_bf16r(v41_fp4_to_f32(nib) * v41_e8m0_to_f32(row[64u + (d >> 5)]));   /* DS4_V41_IDXK_NIB = 64 */
}
// 稀疏注意力: grid (n_tok, n_head/8), 256 线程。键序 = 窗口行(升序) 后接 topk 压缩行; 8 键一块在线 softmax; p 舍 bf16 再乘 v; sink 只进分母。
kernel void kernel_v41_sparse_attn(constant v41_attn_args &a [[buffer(0)]], device float *o [[buffer(1)]], device const float *q [[buffer(2)]],
                                   device const float *kvw [[buffer(3)]], device const uchar *kvc [[buffer(4)]], device const int *idx [[buffer(5)]],
                                   device const float *sink [[buffer(6)]], device const int *posd [[buffer(7)]],
                                   uint2 tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ks[8][512];
    threadgroup float ksc[8][32];
    threadgroup int kok[8];
    const uint i = tg.x, h = tg.y * 8u + sgitg, hd = a.hd, per = hd / 32u;
    const uint pos0 = a.has_posd ? (uint)posd[0] : a.pos0;
    float qa[16], acc[16];
    for (uint e = 0; e < per; e++) { qa[e] = q[((ulong)i * a.n_head + h) * hd + lane * per + e]; acc[e] = 0.0f; }
    float mx = -1e30f, sum = 0.0f;
    const uint p = pos0 + i;
    const uint last = a.full_block ? pos0 + a.full_block - 1u : p;
    uint lo = a.full_block ? (pos0 > a.window ? pos0 - a.window : 0u) : (p + 1u > a.window ? p + 1u - a.window : 0u);
    if (lo < a.win_lo) lo = a.win_lo;
    const uint nwin = last - lo + 1u;
    const uint topk = a.has_comp ? a.topk : 0u;
    const uint nkeys = nwin + topk;
    for (uint base = 0; base < nkeys; base += 8u) {
        const uint nt = (nkeys - base) < 8u ? (nkeys - base) : 8u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < nt) {   /* simdgroup t 装第 base+t 个键 */
            const uint t = sgitg, kk = base + t;
            bool have_w = false, have_c = false; ulong wrow = 0; device const uchar *cpk = kvc;
            if (kk < nwin) { have_w = true; wrow = (ulong)v41_win_row((int)(lo + kk), pos0, a.window, a.ring) * hd; }
            else { const int g = idx[(ulong)i * topk + (kk - nwin)]; if (g >= 0 && (uint)g < a.ng) { have_c = true; cpk = kvc + (ulong)g * 288u; } }
            if (lane == 0u) kok[t] = (have_w || have_c) ? 1 : 0;
            if (have_c) ksc[t][lane] = v41_e4m3_to_f32(cpk[256u + lane]);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = lane; d < hd; d += 32u) ks[t][d] = have_w ? kvw[wrow + d] : (have_c ? v41_ckv_get_s(cpk, d, ksc[t]) : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[8]; float tm = -1e30f;
        for (uint t = 0; t < nt; t++) {
            float d = 0.0f;
            for (uint e = 0; e < per; e++) d += qa[e] * ks[t][lane * per + e];
            d = simd_sum(d);
            s[t] = kok[t] ? d * a.scale : -1e30f;
            tm = max(tm, s[t]);
        }
        const float nm = max(mx, tm), rs = exp(mx - nm);
        float add[16]; for (uint e = 0; e < per; e++) add[e] = 0.0f;
        float ps = 0.0f;
        for (uint t = 0; t < nt; t++) {
            const float pv = exp(s[t] - nm);
            ps += pv;
            const float pb = v41_bf16r(pv);
            for (uint e = 0; e < per; e++) add[e] += pb * ks[t][lane * per + e];
        }
        sum = sum * rs + ps;
        for (uint e = 0; e < per; e++) acc[e] = acc[e] * rs + add[e];
        mx = nm;
    }
    const float den = sum + exp(sink[h] - mx);
    for (uint e = 0; e < per; e++) o[((ulong)i * a.n_head + h) * hd + lane * per + e] = v41_bf16r(acc[e] / den);
}

// ---- indexer 打分(官方 Indexer.forward): score[i][c] = bf16(Σ_h bf16(relu(bf16(q_h·k_g))·w[i][h])); 不可见 → −inf
// 一 simdgroup 一个候选项 c; grid (n_tok, gblocks) × 256 线程。cand 非空(C2): 紧凑行 [ns], 第 c 项 ↔ 组 list[1+c/bs]·bs + c%bs。
struct v41_idx_args { uint pos0, ng, n_head, dk, ratio, has_posd, has_cand, cand_bs, cand_cap, n_rows, topk, pad; };
inline uint v41_cand_ns(uint ng, uint bs, uint cap) { const ulong full = (ulong)cap * bs; return full < ng ? (uint)full : ng; }
kernel void kernel_v41_indexer_score(constant v41_idx_args &a [[buffer(0)]], device float *score [[buffer(1)]], device const float *q [[buffer(2)]],
                                     device const uchar *k [[buffer(3)]], device const float *w [[buffer(4)]], device const int *cand [[buffer(5)]],
                                     device const int *posd [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint2 ntg [[threadgroups_per_grid]]) {
    const uint i = tg.x;
    uint pos0 = a.pos0, ng = a.ng;
    if (a.has_posd) { pos0 = (uint)posd[0]; ng = (pos0 + a.n_rows) / a.ratio; }
    const uint vis = (pos0 + i + 1u) / a.ratio, per = a.dk / 32u;
    device const int *cl = a.has_cand ? cand + (ulong)i * (1u + a.cand_cap) : cand;
    const uint nc = a.has_cand ? (uint)cl[0] : 0u;
    const uint ns = a.has_cand ? v41_cand_ns(ng, a.cand_bs, a.cand_cap) : ng;
    const uint gstride = 8u * ntg.y;
    for (uint c = tg.y * 8u + sgitg; c < ns; c += gstride) {
        uint g = c;
        if (a.has_cand) { const uint b = c / a.cand_bs; g = b < nc ? (uint)cl[1u + b] * a.cand_bs + c % a.cand_bs : ng; }
        const bool live = g < ng && g < vis;
        float kv[4];
        device const uchar *kg = k + (ulong)(live ? g : 0u) * 72u;   /* DS4_V41_IDXK_BYTES */
        for (uint e = 0; e < 4u; e++) kv[e] = (live && e < per) ? v41_idxk_get(kg, lane * per + e) : 0.0f;
        float acc = 0.0f;
        for (uint h = 0; h < a.n_head; h++) {
            device const float *qh = q + ((ulong)i * a.n_head + h) * a.dk;
            float d = 0.0f;
            for (uint e = 0; e < 4u; e++) if (e < per) d += qh[lane * per + e] * kv[e];
            d = simd_sum(d);
            float dd = v41_bf16r(d);
            dd = max(dd, 0.0f);
            acc += v41_bf16r(dd * w[(ulong)i * a.n_head + h]);
        }
        if (lane == 0u) score[(ulong)i * ns + c] = live ? v41_bf16r(acc) : -INFINITY;
    }
}

// ---- 候选块(官方 select_candidate_blocks): 块分 = 块内最大, 含最新位置的块钉 +inf; kk ≥ nb 全选, 否则 radix select;
// 选中的块压成升序列表 list[i][0] = 块数, list[i][1..] = 块号。一 threadgroup 一 query, 256 线程; blk 是全局暂存 [n][nb_cap] 分 + [n][nb_cap] 标记。
struct v41_cand_args { uint pos0, ng, ratio, topk_blocks, bs, has_posd, n_rows, nb_cap; };
kernel void kernel_v41_candidate(constant v41_cand_args &a [[buffer(0)]], device int *list [[buffer(1)]], device const float *score [[buffer(2)]],
                                 device const int *posd [[buffer(3)]], device float *blk [[buffer(4)]],
                                 uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint cnt[256];
    threadgroup uint sh_bucket, sh_k;
    uint pos0 = a.pos0, ng = a.ng;
    if (a.has_posd) { pos0 = (uint)posd[0]; ng = (pos0 + a.n_rows) / a.ratio; }
    const uint nb = (ng + a.bs - 1u) / a.bs, vis = (pos0 + i + 1u) / a.ratio;
    device float *bsc = blk + (ulong)i * a.nb_cap;
    device uchar *sel = (device uchar *)(blk + (ulong)a.n_rows * a.nb_cap) + (ulong)i * a.nb_cap;
    for (uint b = tid; b < nb; b += 256u) {
        float m = -INFINITY;
        for (uint j = b * a.bs; j < (b + 1u) * a.bs && j < ng; j++) m = max(m, score[(ulong)i * ng + j]);
        if (vis > 0u && b == (vis - 1u) / a.bs) m = INFINITY;
        bsc[b] = m; sel[b] = 0;
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint kk = a.topk_blocks < nb ? a.topk_blocks : nb;
    if (kk >= nb) {
        for (uint b = tid; b < nb; b += 256u) sel[b] = v41_topk_key(bsc[b]) != 0u ? 1 : 0;
    } else {
        uint prefix = 0u, want = kk;
        for (int shift = 24; shift >= 0; shift -= 8) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint b = tid; b < nb; b += 256u) {
                const uint key = v41_topk_key(bsc[b]);
                if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
                atomic_fetch_add_explicit(&hist[(key >> shift) & 0xffu], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint acc = 0u; int t = 255;
                for (; t > 0; t--) { const uint hv = atomic_load_explicit(&hist[t], memory_order_relaxed); if (acc + hv >= want) break; acc += hv; }
                sh_bucket = (uint)t; sh_k = want - acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix |= sh_bucket << shift;
            want = sh_k;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint b = tid; b < nb; b += 256u) sel[b] = v41_topk_key(bsc[b]) > prefix ? 1 : 0;
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) {
            uint eq = want;
            for (uint b = 0; b < nb && eq; b++) if (v41_topk_key(bsc[b]) == prefix) { sel[b] = 1; eq--; }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint chunk = (nb + 255u) / 256u;
    const uint c0 = tid * chunk, c1 = (c0 + chunk) < nb ? (c0 + chunk) : nb;
    uint mine = 0u;
    for (uint b = c0; b < c1; b++) mine += sel[b];
    cnt[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint run = 0u;
        for (uint t = 0; t < 256u; t++) { const uint v = cnt[t]; cnt[t] = run; run += v; }
        list[(ulong)i * (1u + a.topk_blocks)] = (int)run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint wr = cnt[tid];
    for (uint b = c0; b < c1; b++) if (sel[b]) list[(ulong)i * (1u + a.topk_blocks) + 1u + wr++] = (int)b;
}

// ---- topk(官方: 取 min(index_topk, 可见组数) 再按位置排序; 不可达 → −1): radix select 阈值 + 并行分段写出(并列取小下标)
// 一 threadgroup 一 query, 256 线程(CUDA 1024 线程 + 8 批读, 整数计数结果与线程数无关)。
inline int v41_cand_map(device const int *cl, uint has_cand, uint c, uint bs) { return has_cand ? cl[1u + c / bs] * (int)bs + (int)(c % bs) : (int)c; }
kernel void kernel_v41_topk(constant v41_idx_args &a [[buffer(0)]], device int *idx [[buffer(1)]], device const float *score [[buffer(2)]],
                            device const int *posd [[buffer(3)]], device const int *cand [[buffer(4)]],
                            uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint suf[257];
    threadgroup atomic_uint sh_bucket, sh_tgt, sh_teq;
    threadgroup uint cgt[256], ceq[256];
    uint ng = a.ng, topk = a.topk;
    if (a.has_posd) { ng = ((uint)posd[0] + a.n_rows) / a.ratio; if (ng < topk) topk = ng; }
    device const int *cl = a.has_cand ? cand + (ulong)i * (1u + a.cand_cap) : cand;
    const uint ns = a.has_cand ? v41_cand_ns(ng, a.cand_bs, a.cand_cap) : ng;
    device const float *s = score + (ulong)i * ns;
    uint prefix = 0u, kk = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        if (tid == 0u) atomic_store_explicit(&sh_bucket, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint g = tid; g < ns; g += 256u) {
            const uint key = v41_topk_key(s[g]);
            if (key == 0u || (shift < 24 && (key >> (shift + 8)) != (prefix >> (shift + 8)))) continue;
            atomic_fetch_add_explicit(&hist[(key >> shift) & 0xffu], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {   /* 后缀和: lane l 管桶 [8l, 8l+8) */
            uint loc = 0u;
            for (uint b = 0; b < 8u; b++) loc += atomic_load_explicit(&hist[tid * 8u + b], memory_order_relaxed);
            uint inc = loc;
            for (uint o = 1u; o < 32u; o <<= 1) { const uint t = simd_shuffle_down(inc, (ushort)o); if (tid + o < 32u) inc += t; }
            uint run = inc - loc;
            for (int b = 7; b >= 0; b--) { run += atomic_load_explicit(&hist[tid * 8u + (uint)b], memory_order_relaxed); suf[tid * 8u + (uint)b] = run; }
            if (tid == 0u) suf[256] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shift == 24) { const uint nval = suf[0]; kk = topk < nval ? topk : nval; if (kk == 0u) break; }
        for (uint b = 1u + tid; b < 256u; b += 256u) if (suf[b] >= kk) atomic_fetch_max_explicit(&sh_bucket, b, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint bs = atomic_load_explicit(&sh_bucket, memory_order_relaxed);
        prefix |= bs << shift;
        kk -= suf[bs + 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { atomic_store_explicit(&sh_tgt, 0u, memory_order_relaxed); atomic_store_explicit(&sh_teq, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chunk = (ns + 255u) / 256u, g0 = tid * chunk, g1 = (g0 + chunk) < ns ? (g0 + chunk) : ns;
    {
        uint ngt = 0u, neq = 0u;
        for (uint g = g0; g < g1; g++) { const uint key = v41_topk_key(s[g]); if (key == 0u) continue; if (key > prefix) ngt++; else if (key == prefix) neq++; }
        cgt[tid] = ngt; ceq[tid] = neq;
        atomic_fetch_add_explicit(&sh_tgt, ngt, memory_order_relaxed); atomic_fetch_add_explicit(&sh_teq, neq, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {   /* 两条排他前缀和(Hillis-Steele) */
        uint vg = tid ? cgt[tid - 1u] : 0u, ve = tid ? ceq[tid - 1u] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cgt[tid] = vg; ceq[tid] = ve;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1u; off < 256u; off <<= 1) {
            const uint ag = tid >= off ? cgt[tid - off] : 0u, ae = tid >= off ? ceq[tid - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cgt[tid] += ag; ceq[tid] += ae;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    {
        const uint eq = kk;
        uint wgt = cgt[tid], weq = ceq[tid];
        for (uint g = g0; g < g1; g++) {
            const uint key = v41_topk_key(s[g]);
            if (key == 0u) continue;
            if (key > prefix) { const uint pos = wgt + (weq < eq ? weq : eq); if (pos < topk) idx[(ulong)i * topk + pos] = v41_cand_map(cl, a.has_cand, g, a.cand_bs); wgt++; }
            else if (key == prefix) { if (weq < eq) { const uint pos = wgt + weq; if (pos < topk) idx[(ulong)i * topk + pos] = v41_cand_map(cl, a.has_cand, g, a.cand_bs); } weq++; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint tg_ = atomic_load_explicit(&sh_tgt, memory_order_relaxed), te = atomic_load_explicit(&sh_teq, memory_order_relaxed);
        uint total = tg_ + (te < kk ? te : kk);
        if (total > topk) total = topk;
        for (uint w = total + tid; w < topk; w += 256u) idx[(ulong)i * topk + w] = -1;
    }
}
