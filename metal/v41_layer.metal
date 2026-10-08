// v41_layer.metal — DeepSeek V4.1 层内小核(Metal, 2026-10-08): RMSNorm / mHC 四件 / engram 门 / 路由 / SwiGLU / 压缩器 /
// RoPE / 激活量化 / KV 打包 / SWA 窗口环。每个核对着 CUDA 同名核(cuda_v41_1,2,3 / cuda_v41_hc / cuda_kv_pack / cuda_kv_ring)逐式写,
// 中间量在哪一步舍 bf16 一个没动。必须排在 v41_common.metal 之后。

// RMSNorm 带权 → bf16: 一 threadgroup 一行, 256 线程
kernel void kernel_v41_rms_norm(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *x [[buffer(2)]],
                                device const float *w [[buffer(3)]], constant ulong &w_off [[buffer(4)]],
                                uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    const uint dim = a.n0; const float eps = a.f0;
    device const float *wr = (device const float *)((device const uchar *)w + w_off);
    device const float *xr = x + (ulong)r * dim; device float *o = out + (ulong)r * dim;
    float s = 0.0f;
    for (uint i = tid; i < dim; i += 256u) s += xr[i] * xr[i];
    const float inv = rsqrt(v41_tg_sum(s, sh, tid, 256u) / float(dim) + eps);
    for (uint i = tid; i < dim; i += 256u) o[i] = v41_bf16r(wr[i] * (xr[i] * inv));
}
// hc_mix 三件: 逐行 1/rms(inv[r]) → f32 GEMV(别处) → mix[r][c] *= inv[r]
kernel void kernel_v41_row_rsqrt(constant v41_n_args &a [[buffer(0)]], device float *inv [[buffer(1)]], device const float *x [[buffer(2)]],
                                 uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    const uint dim = a.n0; const float eps = a.f0;
    device const float *xr = x + (ulong)r * dim;
    float s = 0.0f;
    for (uint i = tid; i < dim; i += 256u) s += xr[i] * xr[i];
    const float t = v41_tg_sum(s, sh, tid, 256u);
    if (tid == 0u) inv[r] = rsqrt(t / float(dim) + eps);
}
kernel void kernel_v41_scale_rows(device float *y [[buffer(0)]], device const float *inv [[buffer(1)]], constant uint &cols [[buffer(2)]],
                                  constant ulong &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i < n) y[i] *= inv[i / cols];
}
// sinkhorn 拆分(官方 hc_split_sinkhorn 逐式): 一 threadgroup 一行, 64 线程, 串行与参考同序
kernel void kernel_v41_hc_split(constant v41_n_args &a [[buffer(0)]], device float *pre [[buffer(1)]], device float *post [[buffer(2)]],
                                device float *comb [[buffer(3)]], device const float *mix [[buffer(4)]], device const float *scale [[buffer(5)]],
                                device const float *base [[buffer(6)]], uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float c[64];
    const uint hc = a.n0, iters = a.n1; const float eps = a.f0;
    const uint mix_hc = 2u * hc + hc * hc;
    device const float *m = mix + (ulong)n * mix_hc;
    if (tid < hc) {
        pre[n * hc + tid] = 1.0f / (1.0f + exp(-(m[tid] * scale[0] + base[tid]))) + eps;
        post[n * hc + tid] = 2.0f / (1.0f + exp(-(m[hc + tid] * scale[1] + base[hc + tid])));
    }
    if (tid < hc * hc) c[tid] = m[2u * hc + tid] * scale[2] + base[2u * hc + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint j = 0; j < hc; j++) {
            float mx = -INFINITY; for (uint k = 0; k < hc; k++) mx = max(mx, c[j * hc + k]);
            float s = 0.0f; for (uint k = 0; k < hc; k++) { c[j * hc + k] = exp(c[j * hc + k] - mx); s += c[j * hc + k]; }
            for (uint k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
        }
        for (uint k = 0; k < hc; k++) {
            float s = 0.0f; for (uint j = 0; j < hc; j++) s += c[j * hc + k];
            for (uint j = 0; j < hc; j++) c[j * hc + k] /= (s + eps);
        }
        for (uint it = 1; it < iters; it++) {
            for (uint j = 0; j < hc; j++) { float s = 0.0f; for (uint k = 0; k < hc; k++) s += c[j * hc + k]; for (uint k = 0; k < hc; k++) c[j * hc + k] /= (s + eps); }
            for (uint k = 0; k < hc; k++) { float s = 0.0f; for (uint j = 0; j < hc; j++) s += c[j * hc + k]; for (uint j = 0; j < hc; j++) c[j * hc + k] /= (s + eps); }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hc * hc) comb[(ulong)n * hc * hc + tid] = c[tid];
}
// hc_pre: out[n][d] = bf16(Σ_c pre[n][c]·hc[n][c][d])
kernel void kernel_v41_hc_pre(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *hc [[buffer(2)]],
                              device const float *pre [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint n_embd = a.n0, n_hc = a.n1, d = gid.x, n = gid.y;
    if (d >= n_embd) return;
    float s = 0.0f;
    for (uint c = 0; c < n_hc; c++) s += pre[n * n_hc + c] * hc[((ulong)n * n_hc + c) * n_embd + d];
    out[(ulong)n * n_embd + d] = v41_bf16r(s);
}
// hc_split + hc_pre + rms_norm 合一(CUDA v41_hc_fused_kernel 的直译): 一 threadgroup 一行, 1024 线程; sinkhorn 在 simdgroup 0 里并行做
// (一线程一元素, 行和/列和线程内按升序串行累加 ⇒ 与串行版逐元素同序)。pre_in = 上一层传下来的 pre(hc_pre 用它, 别用成本层的 pre)。
kernel void kernel_v41_hc_fused(constant v41_n_args &a [[buffer(0)]], device float *pre [[buffer(1)]], device float *post [[buffer(2)]],
                                device float *comb [[buffer(3)]], device float *x [[buffer(4)]], device float *xn [[buffer(5)]],
                                device const float *mix [[buffer(6)]], device const float *hc [[buffer(7)]], device const float *pre_in [[buffer(8)]],
                                device const float *scale [[buffer(9)]], device const float *base [[buffer(10)]], device const float *nw [[buffer(11)]],
                                uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float c[64], sh[1024];
    const uint n_embd = a.n0, n_hc = a.n1, iters = a.n2; const float hc_eps = a.f0, norm_eps = a.f1;
    const uint mix_hc = 2u * n_hc + n_hc * n_hc;
    device const float *m = mix + (ulong)n * mix_hc;
    if (sgitg == 0u) {
        if (tid < n_hc) {
            pre[n * n_hc + tid] = 1.0f / (1.0f + exp(-(m[tid] * scale[0] + base[tid]))) + hc_eps;
            post[n * n_hc + tid] = 2.0f / (1.0f + exp(-(m[n_hc + tid] * scale[1] + base[n_hc + tid])));
        }
        if (tid < n_hc * n_hc) c[tid] = m[2u * n_hc + tid] * scale[2] + base[2u * n_hc + tid];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint tj = tid / n_hc, tk = tid % n_hc;
        const bool act = tid < n_hc * n_hc;
        float nv = 0.0f;
        if (act) {   /* ① 行 softmax + eps */
            float mx = -INFINITY;
            for (uint k = 0; k < n_hc; k++) mx = max(mx, c[tj * n_hc + k]);
            float sum = 0.0f;
            for (uint k = 0; k < n_hc; k++) sum += exp(c[tj * n_hc + k] - mx);
            nv = exp(c[tj * n_hc + tk] - mx) / sum + hc_eps;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (act) c[tj * n_hc + tk] = nv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (act) { float cs = 0.0f; for (uint j = 0; j < n_hc; j++) cs += c[j * n_hc + tk]; nv = c[tj * n_hc + tk] / (cs + hc_eps); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (act) c[tj * n_hc + tk] = nv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint it = 1; it < iters; it++) {
            if (act) { float rs = 0.0f; for (uint k = 0; k < n_hc; k++) rs += c[tj * n_hc + k]; nv = c[tj * n_hc + tk] / (rs + hc_eps); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (act) c[tj * n_hc + tk] = nv;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (act) { float cs = 0.0f; for (uint j = 0; j < n_hc; j++) cs += c[j * n_hc + tk]; nv = c[tj * n_hc + tk] / (cs + hc_eps); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (act) c[tj * n_hc + tk] = nv;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (act) comb[(ulong)n * n_hc * n_hc + tid] = c[tid];
    }
    /* hc_pre → x, 顺手攒 x² 给 rms(1024 槽树, 与 CUDA 1024 线程版同形状) */
    float acc = 0.0f;
    device const float *hcn = hc + (ulong)n * n_hc * n_embd, *pin = pre_in + (ulong)n * n_hc;
    device float *xr = x + (ulong)n * n_embd, *on = xn + (ulong)n * n_embd;
    for (uint d = tid; d < n_embd; d += 1024u) {
        float v = 0.0f;
        for (uint k = 0; k < n_hc; k++) v += pin[k] * hcn[(ulong)k * n_embd + d];
        v = v41_bf16r(v);
        xr[d] = v;
        acc += v * v;
    }
    const float inv = rsqrt(v41_tg_sum(acc, sh, tid, 1024u) / float(n_embd) + norm_eps);
    for (uint d = tid; d < n_embd; d += 1024u) on[d] = v41_bf16r(nw[d] * (xr[d] * inv));
}
// hc_post: out[k][d] = bf16(post[k]·y[d] + Σ_j comb[j][k]·res[j][d])
kernel void kernel_v41_hc_post(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *y [[buffer(2)]],
                               device const float *res [[buffer(3)]], device const float *post [[buffer(4)]], device const float *comb [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint n_embd = a.n0, n_hc = a.n1, d = gid.x, n = gid.y;
    if (d >= n_embd) return;
    const float yv = y[(ulong)n * n_embd + d];
    for (uint k = 0; k < n_hc; k++) {
        float s = post[n * n_hc + k] * yv;
        for (uint j = 0; j < n_hc; j++) s += comb[(ulong)n * n_hc * n_hc + j * n_hc + k] * res[((ulong)n * n_hc + j) * n_embd + d];
        out[((ulong)n * n_hc + k) * n_embd + d] = v41_bf16r(s);
    }
}
// engram 门(官方 Engram.forward 后半): 一 threadgroup 一 (token, 路), 256 线程归约 h²、key²、h·w·key
kernel void kernel_v41_engram_gate(constant v41_n_args &a [[buffer(0)]], device float *hc [[buffer(1)]], device const float *kv [[buffer(2)]],
                                   device const float *qw [[buffer(3)]], device const float *kw [[buffer(4)]],
                                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    const uint E = a.n0, n_hc = a.n1, c = tg.x, t = tg.y; const float eps = a.f0;
    device float *h = hc + ((ulong)t * n_hc + c) * E;
    device const float *key = kv + (ulong)t * (n_hc + 1u) * E + (ulong)c * E;
    device const float *val = kv + (ulong)t * (n_hc + 1u) * E + (ulong)n_hc * E;
    float sh_ = 0.0f, sk = 0.0f, sd = 0.0f;
    for (uint d = tid; d < E; d += 256u) {
        const float hv = h[d], kv_ = key[d], w = qw[c * E + d] * kw[c * E + d];
        sh_ += hv * hv; sk += kv_ * kv_; sd += hv * w * kv_;
    }
    const float r0 = v41_tg_sum(sh_, sh, tid, 256u), r1 = v41_tg_sum(sk, sh, tid, 256u), r2 = v41_tg_sum(sd, sh, tid, 256u);
    const float rstd = rsqrt(r0 / float(E) + eps) * rsqrt(r1 / float(E) + eps);
    const float dot = r2 * rstd * rsqrt(float(E));
    const float mag = sqrt(max(abs(dot), 1e-6f));
    const float z = copysign(mag, dot);
    const float gate = 1.0f / (1.0f + exp(-z));
    for (uint d = tid; d < E; d += 256u) h[d] = v41_bf16r(h[d] + gate * val[d]);
}
// 路由(官方 Gate: probs = √softplus; 按 probs+bias 选 topk; weights = probs/Σ·route_scale): 一 simdgroup 一 token, lane 持 12 个专家
struct v41_router_args { uint n_tok, n_expert, topk, pad; float route_scale; float pad1, pad2, pad3; };
kernel void kernel_v41_router(constant v41_router_args &a [[buffer(0)]], device int *sel [[buffer(1)]], device float *wts [[buffer(2)]],
                              device const float *logits [[buffer(3)]], device const float *bias [[buffer(4)]],
                              uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint t = gid >> 5;
    if (t >= a.n_tok) return;
    device const float *lg = logits + (ulong)t * a.n_expert;
    float pr[12], sc[12];
    for (uint k = 0; k < 12u; k++) {
        const uint e = lane + 32u * k;
        if (e < a.n_expert) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : log(1.0f + exp(z));   /* softplus(β=1, threshold 20 与 torch 同) */
            pr[k] = sqrt(sp); sc[k] = pr[k] + bias[e];
        } else { pr[k] = 0.0f; sc[k] = -INFINITY; }
    }
    uint used = 0u; float wsum = 0.0f;
    for (uint r = 0; r < a.topk; r++) {
        float bv = -INFINITY; int bk = -1;
        for (uint k = 0; k < 12u; k++) if (!((used >> k) & 1u) && sc[k] > bv) { bv = sc[k]; bk = (int)k; }
        int be = bk >= 0 ? (int)(lane + 32u * (uint)bk) : -1;
        for (uint off = 16u; off > 0u; off >>= 1) {
            const float ov = simd_shuffle_xor(bv, (ushort)off); const int oe = simd_shuffle_xor(be, (ushort)off);
            if (oe >= 0 && (be < 0 || ov > bv || (ov == bv && oe < be))) { bv = ov; be = oe; }
        }
        float myp = 0.0f;
        if (be >= 0 && (uint)(be & 31) == lane) { const uint k = (uint)be >> 5; used |= 1u << k; myp = pr[k]; }
        myp = simd_sum(myp);   /* 只有一个 lane 非零 */
        if (lane == 0u) { sel[(ulong)t * a.topk + r] = be; wts[(ulong)t * a.topk + r] = myp; }
        wsum += myp;
    }
    if (lane == 0u) for (uint r = 0; r < a.topk; r++) wts[(ulong)t * a.topk + r] = wts[(ulong)t * a.topk + r] / (wsum + 1e-20f) * a.route_scale;
}
// SwiGLU(官方 Expert 截断语义): h = bf16(silu(min(g, L))·clamp(u, ±L))
kernel void kernel_v41_swiglu(device float *h [[buffer(0)]], device const float *g [[buffer(1)]], device const float *u [[buffer(2)]],
                              constant ulong &n [[buffer(3)]], constant float &limit [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) { uv = min(max(uv, -limit), limit); gv = min(gv, limit); }
    h[i] = v41_bf16r((gv / (1.0f + exp(-gv))) * uv);
}
// 压缩器池化(ratio>1): 组内逐维 softmax 加权和 → bf16。一 threadgroup 一组, 256 线程
kernel void kernel_v41_compress_pool(constant v41_n_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *kv [[buffer(2)]],
                                     device const float *sc [[buffer(3)]], uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint ratio = a.n0, dim = a.n1;
    for (uint d = tid; d < dim; d += 256u) {
        float mx = -INFINITY;
        for (uint t = 0; t < ratio; t++) mx = max(mx, sc[((ulong)g * ratio + t) * dim + d]);
        float den = 0.0f, acc = 0.0f;
        for (uint t = 0; t < ratio; t++) {
            const float e = exp(sc[((ulong)g * ratio + t) * dim + d] - mx);
            den += e; acc += e * kv[((ulong)g * ratio + t) * dim + d];
        }
        out[(ulong)g * dim + d] = v41_bf16r(acc / den);
    }
}
// 压缩源层的解码一步(n ≤ 8 行): 逐行追加到余行缓冲第 (pos0%ratio + i)%ratio 格, 凑满一组池化一次; 池化与上面逐位同
struct v41_cstep_args { uint ratio, dim, n, has_snap; };
kernel void kernel_v41_compress_step_n(constant v41_cstep_args &a [[buffer(0)]], device float *pooled [[buffer(1)]], device int *posg [[buffer(2)]],
                                       device float *ckv_c [[buffer(3)]], device float *csc_c [[buffer(4)]], device float *snap_kv [[buffer(5)]],
                                       device float *snap_sc [[buffer(6)]], device const float *ckv [[buffer(7)]], device const float *csc [[buffer(8)]],
                                       device const int *posd [[buffer(9)]], uint tid [[thread_index_in_threadgroup]]) {
    const uint ratio = a.ratio, dim = a.dim, n = a.n, pos0 = (uint)posd[0], pend = pos0 % ratio;
    if (a.has_snap)
        for (uint r = 0; r < pend; r++)
            for (uint d = tid; d < dim; d += 256u) { snap_kv[(ulong)r * dim + d] = ckv_c[(ulong)r * dim + d]; snap_sc[(ulong)r * dim + d] = csc_c[(ulong)r * dim + d]; }
    uint j = 0;
    for (uint i = 0; i < n; i++) {
        const uint slot = (pend + i) % ratio;
        threadgroup_barrier(mem_flags::mem_device);
        for (uint d = tid; d < dim; d += 256u) {
            const float av = ckv[(ulong)i * dim + d], bv = csc[(ulong)i * dim + d];
            ckv_c[(ulong)slot * dim + d] = av; csc_c[(ulong)slot * dim + d] = bv;
            if (a.has_snap) { snap_kv[(ulong)(pend + i) * dim + d] = av; snap_sc[(ulong)(pend + i) * dim + d] = bv; }
        }
        if (slot + 1u != ratio) continue;
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) posg[j] = (int)(pos0 + i + 1u - ratio);
        for (uint d = tid; d < dim; d += 256u) {
            float mx = -INFINITY;
            for (uint t = 0; t < ratio; t++) mx = max(mx, csc_c[(ulong)t * dim + d]);
            float den = 0.0f, acc = 0.0f;
            for (uint t = 0; t < ratio; t++) { const float e = exp(csc_c[(ulong)t * dim + d] - mx); den += e; acc += e * ckv_c[(ulong)t * dim + d]; }
            pooled[(ulong)j * dim + d] = v41_bf16r(acc / den);
        }
        j++;
    }
}
// RoPE(官方 precompute_freqs_cis + apply_rotary_emb; YaRN ramp): grid (n_head, n_tok) × 64 线程(复数下标)。round=1 舍 bf16(前向), 0 不舍(反向)
struct v41_rope_args { uint n_head, head_dim, n_rot, osl, inverse, do_round; float theta, factor, beta_fast, beta_slow; };
inline float v41_rope_freq(uint i, uint dim, float theta, uint osl, float factor, float beta_fast, float beta_slow) {
    float f = 1.0f / pow(theta, float(2u * i) / float(dim));
    if (osl > 0u) {
        const float lt = 2.0f * log(theta);
        float lo = floor(float(dim) * log(float(osl) / (beta_fast * 2.0f * M_PI_F)) / lt);
        float hi = ceil(float(dim) * log(float(osl) / (beta_slow * 2.0f * M_PI_F)) / lt);
        lo = max(lo, 0.0f); hi = min(hi, float(dim - 1u));
        float ramp = (float(i) - lo) / max(hi - lo, 1e-3f);
        ramp = min(max(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        f = f / factor * (1.0f - smooth) + f * smooth;
    }
    return f;
}
kernel void kernel_v41_rope(constant v41_rope_args &a [[buffer(0)]], device float *x [[buffer(1)]], device const int *pos [[buffer(2)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, t = tg.y;
    if (i >= a.n_rot / 2u) return;
    device float *xr = x + ((ulong)t * a.n_head + h) * a.head_dim + (a.head_dim - a.n_rot) + 2u * i;
    const float ang = float(pos[t]) * v41_rope_freq(i, a.n_rot, a.theta, a.osl, a.factor, a.beta_fast, a.beta_slow);
    float c = cos(ang), s = sin(ang);
    if (a.inverse) s = -s;
    const float p = xr[0], q = xr[1];
    const float r0 = p * c - q * s, r1 = p * s + q * c;
    xr[0] = a.do_round ? v41_bf16r(r0) : r0;
    xr[1] = a.do_round ? v41_bf16r(r1) : r1;
}
// 激活量化就地: 一 simdgroup 一个块(≤32 元素)。mode 0: fp8 e4m3 + ue8m0(448); 1: fp4 + ue8m0(6); 2: fp4 + e4m3 缩放
kernel void kernel_v41_act_quant(constant v41_n_args &a [[buffer(0)]], device float *x [[buffer(1)]], uint gid [[thread_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint dim = a.n0, block = a.n1, mode = a.n2, nb_row = a.n3, blk = gid >> 5, row = blk / nb_row, b = blk % nb_row;
    device float *p = x + (ulong)row * dim + (ulong)b * block;
    const float v = lane < block ? p[lane] : 0.0f;
    float am = simd_max(abs(v));
    float s;
    if (mode == 0u) { am = max(am, 1e-4f); s = v41_pow2_ceil_log2(am / 448.0f); }
    else if (mode == 1u) { am = max(am, 6.0f * ldexp(1.0f, -126)); s = v41_pow2_ceil_log2(am / 6.0f); }
    else { am = max(am, 6.0f * ldexp(1.0f, -9)); s = v41_e4m3_round(am / 6.0f); }
    if (lane < block) {
        float q = v / s;
        if (mode == 0u) { q = min(max(q, -448.0f), 448.0f); q = v41_e4m3_round(q); }
        else { q = min(max(q, -6.0f), 6.0f); q = v41_e2m1_round(q); }
        p[lane] = v41_bf16r(q * s);
    }
}
// KV 打包(与 act_quant 的分块/求最大/定缩放三步逐字同): mode 0 主 KV(块 16, E4M3 缩放) / 1 索引 K(块 32, E8M0 缩放)
struct v41_kvpack_args { uint g0, dim, blk, row_bytes, nib_bytes, nb_row, mode, has_posd, ratio, g_trash, nbatch, pad; };
kernel void kernel_v41_kv_pack(constant v41_kvpack_args &a [[buffer(0)]], device uchar *cache [[buffer(1)]], device const float *rows [[buffer(2)]],
                               device const int *posd [[buffer(3)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint sg = gid >> 5, r = sg / a.nb_row, b = sg % a.nb_row;
    uint grp = a.g0 + r;
    if (a.has_posd) {
        const uint pos = (uint)posd[0], ng_new = (pos % a.ratio + a.nbatch) / a.ratio;
        grp = r < ng_new ? pos / a.ratio + r : a.g_trash;
    }
    device const float *p = rows + (ulong)r * a.dim + (ulong)b * a.blk;
    const float v = lane < a.blk ? p[lane] : 0.0f;
    float am = simd_max(abs(v));
    float s;
    if (a.mode == 0u) { am = max(am, 6.0f * ldexp(1.0f, -9)); s = v41_e4m3_round(am / 6.0f); }
    else { am = max(am, 6.0f * ldexp(1.0f, -126)); s = v41_pow2_ceil_log2(am / 6.0f); }
    device uchar *row = cache + (ulong)grp * a.row_bytes;
    if (lane == 0u) row[a.nib_bytes + b] = (uchar)(a.mode == 0u ? v41_e4m3_f32_to_byte(s) : v41_e8m0_f32_to_byte(s));
    const uint nib = lane < a.blk ? v41_fp4_f32_to_nibble(v / s) : 0u;
    const uint odd = simd_shuffle_down(nib, 1u);
    if (lane < a.blk && (lane & 1u) == 0u) row[(b * a.blk + lane) >> 1] = (uchar)(nib | (odd << 4));
}
// SWA 窗口环: commit(本批第 i 行 → 环格 (pos0+i)%window) / snap(back=0 存, 1 回写)
struct v41_ring_args { uint pos0, i0, window, hd, back, has_posd, pad0, pad1; };
kernel void kernel_v41_win_commit(constant v41_ring_args &a [[buffer(0)]], device float *win [[buffer(1)]], device const int *posd [[buffer(2)]],
                                  uint j [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint i = a.i0 + j, pos0 = a.has_posd ? (uint)posd[0] : a.pos0;
    device const float *src = win + (ulong)(a.window + i) * a.hd;
    device float *dst = win + (ulong)((pos0 + i) % a.window) * a.hd;
    for (uint d = tid; d < a.hd; d += 256u) dst[d] = src[d];
}
kernel void kernel_v41_win_snap(constant v41_ring_args &a [[buffer(0)]], device float *win [[buffer(1)]], device float *snap [[buffer(2)]],
                                device const int *posd [[buffer(3)]], uint j [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint i = a.i0 + j, pos0 = a.has_posd ? (uint)posd[0] : a.pos0;
    device float *ring = win + (ulong)((pos0 + i) % a.window) * a.hd;
    device float *sp = snap + (ulong)i * a.hd;
    if (a.back) { for (uint d = tid; d < a.hd; d += 256u) ring[d] = sp[d]; }
    else { for (uint d = tid; d < a.hd; d += 256u) sp[d] = ring[d]; }
}
