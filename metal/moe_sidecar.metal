// staged-RHS and 32-token variants were benchmark-only experiments.
template<short NR1>
kernel void kernel_attn_out_low_q8_0_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne21;
    const int G = args.ne1;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;
    const bool full_tile = r0 + NR0 <= M && r1 + NR1 <= N && (K % NK) == 0;

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device float *ptrB = (device float *)(srcB + args.nb11*group);
    const int strideB = args.nb12/sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (full_tile || r0 + row < M) {
                const int block_idx = k_pos/32;
                const short il = (k_pos/16)%2;
                device const block_q8_0 *row_ptr =
                    (device const block_q8_0 *)(srcA + args.nb01*(r0 + row) + group*args.nb02);

                half4x4 temp_a;
                dequantize_q8_0(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (full_tile || k_pos + i < K) ? temp_a[i/4][i%4] : (half)0;
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_group = (device float *)dst + group*M;
    if (full_tile) {
        device float *dst_tile = dst_group + r0 + (uint64_t)r1*G*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, G*M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_group, dextents<int32_t, 2>(M, N), array<int, 2>({1, G*M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_attn_out_low_q8_0_mpp_direct_rhs<64>) attn_out_low_q8_0_mpp_direct_rhs_n64_t;

template [[host_name("kernel_attn_out_low_q8_0_mpp_direct_rhs_n64")]] kernel attn_out_low_q8_0_mpp_direct_rhs_n64_t kernel_attn_out_low_q8_0_mpp_direct_rhs<64>;

#endif

// ---------------------------------------------------------------------------
// go1b "hidden variable z^L" four-loss correction (sidecar corr_* tensors).
//
// The routed-MoE forward emits the strict-1-bit (go1b) expert output o_hat_e(x).
// A tiny resident low-rank correction is added on top to recover the full-precision
// behaviour without storing the FP experts:
//
//   router_logit[e] += delta[e]                          (before top-k selection)
//   out += sum over selected e of  U @ (C[e] (.*) (V @ x)) + b + beta[e]
//
// d_l (the low-rank width) is small (~64), so V@x and U@t are cheap per-token
// matvecs.  x is the per-token post-RMSNorm FFN activation fed to the experts.
// ---------------------------------------------------------------------------

struct ds4_metal_corr_router_bias_args {
    uint32_t n_expert;
    uint32_t n_tokens;
};

// logits[t][e] += delta[e], broadcast over tokens.  Applied to the raw router
// logits before softplus/sqrt + top-k (score-routed layers only).
kernel void kernel_dsv4_corr_router_bias(
        constant ds4_metal_corr_router_bias_args &args,
        device float *logits,            // [n_tokens][n_expert] in/out
        device const float *delta,       // [n_expert]
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.n_expert;
    if (gid >= total) return;
    const uint e = gid % args.n_expert;
    logits[gid] += delta[e];
}

struct ds4_metal_corr_apply_args {
    uint32_t d_model;
    uint32_t d_l;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
};

// One threadgroup per token.  Phase 1 cooperatively builds vx[i] = (V @ x)[i] in
// threadgroup memory; phase 2 accumulates the per-selected-expert correction into
// out[t][d].  b is added once per selected expert and beta[e] is a per-expert
// scalar, matching  o_e += U @ (C[e] .* vx) + b + beta[e]  for each selected e.
kernel void kernel_dsv4_corr_apply(
        constant ds4_metal_corr_apply_args &args,
        device float *out,               // [n_tokens][d_model] in/out (+= correction)
        device const float *x,           // [n_tokens][d_model]
        device const float *U,           // [d_model][d_l] row-major
        device const float *V,           // [d_l][d_model] row-major
        device const float *C,           // [n_expert][d_l] row-major
        device const float *b,           // [d_model]
        device const float *beta,        // [n_expert]
        device const int   *selected,    // [n_tokens][n_expert_used]
        threadgroup float  *vx [[threadgroup(0)]],   // [d_l]
        uint tok [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (tok >= args.n_tokens) return;
    const uint d_model = args.d_model;
    const uint d_l     = args.d_l;
    const uint n_sel   = args.n_expert_used;
    const uint n_exp   = args.n_expert;

    device const float *xt  = x + (uint64_t)tok * d_model;
    device const int   *sel = selected + (uint64_t)tok * n_sel;

    // Phase 1: vx[i] = sum_j V[i][j] * x[j]
    for (uint i = tid; i < d_l; i += ntg) {
        device const float *Vrow = V + (uint64_t)i * d_model;
        float acc = 0.0f;
        for (uint j = 0; j < d_model; j++) acc += Vrow[j] * xt[j];
        vx[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum of beta over selected experts (scalar contribution, same for every d)
    float beta_sum = 0.0f;
    uint  n_valid  = 0u;
    for (uint s = 0; s < n_sel; s++) {
        const int e = sel[s];
        if (e >= 0 && (uint)e < n_exp) { beta_sum += beta[e]; n_valid++; }
    }

    // Phase 2: out[t][d] += sum_e (U[d] . (C[e] .* vx)) + n_valid*b[d] + beta_sum
    device float *outt = out + (uint64_t)tok * d_model;
    for (uint d = tid; d < d_model; d += ntg) {
        device const float *Urow = U + (uint64_t)d * d_l;
        float acc = 0.0f;
        for (uint s = 0; s < n_sel; s++) {
            const int e = sel[s];
            if (e < 0 || (uint)e >= n_exp) continue;
            device const float *Crow = C + (uint64_t)e * d_l;
            float p = 0.0f;
            for (uint i = 0; i < d_l; i++) p += Urow[i] * Crow[i] * vx[i];
            acc += p;
        }
        acc += (float)n_valid * b[d] + beta_sum;
        outt[d] += acc;
    }
}

// Store variant: writes the raw correction term into a SEPARATE delta buffer
// (out[t][d] = corr term) instead of accumulating into routed_out. The decode
// consumer (shared-down HC fusion) adds it as routed[d]+delta[d] — same two
// operands, same single fadd as the in-place kernel, so results stay
// bit-identical while routed_out carries no read/write hazard from this tiny
// dispatch (the R/W self-alias was a full pipeline-drain bubble per layer).
kernel void kernel_dsv4_corr_delta(
        constant ds4_metal_corr_apply_args &args,
        device float *out,               // [n_tokens][d_model] delta (overwritten)
        device const float *x,           // [n_tokens][d_model]
        device const float *U,           // [d_model][d_l] row-major
        device const float *V,           // [d_l][d_model] row-major
        device const float *C,           // [n_expert][d_l] row-major
        device const float *b,           // [d_model]
        device const float *beta,        // [n_expert]
        device const int   *selected,    // [n_tokens][n_expert_used]
        threadgroup float  *vx [[threadgroup(0)]],   // [d_l]
        uint tok [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (tok >= args.n_tokens) return;
    const uint d_model = args.d_model;
    const uint d_l     = args.d_l;
    const uint n_sel   = args.n_expert_used;
    const uint n_exp   = args.n_expert;

    device const float *xt  = x + (uint64_t)tok * d_model;
    device const int   *sel = selected + (uint64_t)tok * n_sel;

    for (uint i = tid; i < d_l; i += ntg) {
        device const float *Vrow = V + (uint64_t)i * d_model;
        float acc = 0.0f;
        for (uint j = 0; j < d_model; j++) acc += Vrow[j] * xt[j];
        vx[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float beta_sum = 0.0f;
    uint  n_valid  = 0u;
    for (uint s = 0; s < n_sel; s++) {
        const int e = sel[s];
        if (e >= 0 && (uint)e < n_exp) { beta_sum += beta[e]; n_valid++; }
    }

    device float *outt = out + (uint64_t)tok * d_model;
    for (uint d = tid; d < d_model; d += ntg) {
        device const float *Urow = U + (uint64_t)d * d_l;
        float acc = 0.0f;
        for (uint s = 0; s < n_sel; s++) {
            const int e = sel[s];
            if (e < 0 || (uint)e >= n_exp) continue;
            device const float *Crow = C + (uint64_t)e * d_l;
            float p = 0.0f;
            for (uint i = 0; i < d_l; i++) p += Urow[i] * Crow[i] * vx[i];
            acc += p;
        }
        acc += (float)n_valid * b[d] + beta_sum;
        outt[d] = acc;
    }
}

#undef QK_NL
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq2xxs_grid
#undef QK_K
#undef N_R0_Q2_K
#undef N_R0_Q4_K
#undef N_R0_IQ2_XXS

// ---------------------------------------------------------------------------
// go-onebit DQZ2 zchain: the quantizer's multiplicative correction chain,
// replayed at inference (math contract: ds4_zchain.h).
//
//   kernel_dsv4_zchain_ge     weights[t][k] *= ge[layer_base + selected[t][k]]
//                             (before the routed matvec; original expert ids)
//   kernel_dsv4_zchain_scale  routed[t][:] *= λ(x[t]) where λ folds the layer's
//                             op list: GL λ<-g·λ | dyn2 λ<-c·λ (c from ‖x‖2) |
//                             dyn8 λ<-c·λ (c from V8·x) | TREF λ<-1+t·(λ-1),
//                             all clamped to [0.25,4] per dyn op — exactly the
//                             quantizer bytes_moe() replay collapsed to a scalar.
//
// Op record: 16 floats — [0]=type, [1]=g/t, [2..5]=w2p, [6..14]=w8,
// [15]=dyn8 V8 block index into the concatenated fp16 [n_blk][8][d_model]
// buffer (-1 = none; V8-less dyn8 records are dropped at load).
// ---------------------------------------------------------------------------

struct ds4_metal_zchain_ge_args {
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t ge_base;        // layer * n_expert
};

kernel void kernel_dsv4_zchain_ge(
        constant ds4_metal_zchain_ge_args &args,
        device float *weights,           // [n_tokens][n_expert_used] in/out
        device const int *selected,      // [n_tokens][n_expert_used] original ids
        device const float *ge,          // [n_layer][n_expert]
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.n_expert_used;
    if (gid >= total) return;
    const int e = selected[gid];
    if (e < 0 || (uint)e >= args.n_expert) return;
    weights[gid] *= ge[args.ge_base + (uint)e];
}

struct ds4_metal_zchain_scale_args {
    uint32_t d_model;
    uint32_t n_tokens;
    uint32_t op_start;       // this layer's range in the packed op table
    uint32_t op_count;
    uint32_t zl_k;           // frozen z^L rank (0 = absent) — type 6, 2026-07-14
    uint32_t zl_off;         // this layer's offset (halves) into the packed zlm table
    uint32_t zl_din;         // V input dim: d=linear | 3d=ftA feature lift (md86)
    float    zl_tr;          // trust-region cap factor
};

// Threadgroup tree-reduce; ntg is a power of two (wrapper guarantees).
static inline float zc_tg_reduce_add(float v, threadgroup float *red, uint tid, uint ntg) {
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// One threadgroup per token. All threads walk the same op records (uniform
// control flow), cooperating on the ‖x‖ / V8·x reductions, then scale the
// token's routed row by the folded λ.
kernel void kernel_dsv4_zchain_scale(
        constant ds4_metal_zchain_scale_args &args,
        device float *routed,            // [n_tokens][d_model] in/out
        device const float *x,           // [n_tokens][d_model]
        device const float *ops,         // [n_ops_total][16]
        device const half *v8,           // [n_blk][8][d_model]
        device const half *zlm,          // packed z^L: per layer z[k] | U[d*k] | V[d*k]
        threadgroup float *red [[threadgroup(0)]],   // [ntg]
        uint tok [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (tok >= args.n_tokens) return;
    const uint d = args.d_model;
    device const float *xt = x + (uint64_t)tok * d;

    // ‖x‖2 once (cheap; used by any dyn2 op)
    float acc = 0.0f;
    for (uint j = tid; j < d; j += ntg) acc += xt[j] * xt[j];
    const float xnorm = sqrt(zc_tg_reduce_add(acc, red, tid, ntg));

    float lam = 1.0f;
    for (uint oi = 0; oi < args.op_count; oi++) {
        device const float *op = ops + (uint64_t)(args.op_start + oi) * 16u;
        const uint ty = (uint)op[0];
        if (ty == 1u) {
            lam = op[1] * lam;
        } else if (ty == 2u) {
            float c = op[2] + op[3] * ((xnorm - op[4]) / op[5]);
            c = clamp(c, 0.25f, 4.0f);
            lam = c * lam;
        } else if (ty == 3u) {
            const int blk = (int)op[15];
            if (blk >= 0) {
                float c = op[6];
                for (uint k = 0; k < 8u; k++) {
                    device const half *vr = v8 + ((uint64_t)blk * 8u + k) * d;
                    float dk = 0.0f;
                    for (uint j = tid; j < d; j += ntg) dk += xt[j] * (float)vr[j];
                    const float dot = zc_tg_reduce_add(dk, red, tid, ntg);
                    c += op[7 + k] * dot;
                }
                c = clamp(c, 0.25f, 4.0f);
                lam = c * lam;
            }
        } else if (ty == 4u) {
            lam = 1.0f + op[1] * (lam - 1.0f);
        }
    }

    device float *rt = routed + (uint64_t)tok * d;
    for (uint j = tid; j < d; j += ntg) rt[j] *= lam;

    // frozen z^L (type 6): routed += clip * U diag(z) V^T x, after the λ scale
    // (record-order parity with the quantizer's bytes_moe; clip base = routed norm).
    if (args.zl_k > 0u) {
        const uint zk = args.zl_k;                    // <= 1024 (enforced at load)
        device const half *hz = zlm + args.zl_off;
        device const half *hU = hz + zk;
        device const half *hV = hU + (uint64_t)d * zk;
        // pv 挪 threadgroup(k>16 时寄存器放不下); c 归属制: 每线程认领 c 子集算完整
        // 点积, 免去逐 c 树归约(k=1024 时归约 barrier 是主开销)。
        threadgroup float pvS[1024];
        const uint din = (args.zl_din > 0u) ? args.zl_din : d;
        float nrm = 0.0f;
        if (din == 3u * d) {   // md86 ftA: φ=[x, x⊙x/rms, relu(x)], rms=sqrt(mean(x²))+1e-6
            float ss_p = 0.0f;
            for (uint j = tid; j < d; j += ntg) ss_p += xt[j] * xt[j];
            const float ss = zc_tg_reduce_add(ss_p, red, tid, ntg);
            nrm = sqrt(ss / (float)d) + 1e-6f;
        }
        for (uint c = tid; c < zk; c += ntg) {
            float dk = 0.0f;
            if (din == 3u * d) {
                for (uint j = 0; j < d; j++) {
                    const float xv = xt[j];
                    dk += xv * (float)hV[(uint64_t)j * zk + c];
                    dk += (xv * xv / nrm) * (float)hV[(uint64_t)(d + j) * zk + c];
                    dk += (xv > 0.0f ? xv : 0.0f) * (float)hV[(uint64_t)(2u * d + j) * zk + c];
                }
            } else {
                for (uint j = 0; j < d; j++) dk += xt[j] * (float)hV[(uint64_t)j * zk + c];
            }
            pvS[c] = dk * (float)hz[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float nd_p = 0.0f, nr_p = 0.0f;
        for (uint j = tid; j < d; j += ntg) {
            float a = 0.0f;
            device const half *ur = hU + (uint64_t)j * zk;
            for (uint c = 0; c < zk; c++) a += pvS[c] * (float)ur[c];
            nd_p += a * a;
            nr_p += rt[j] * rt[j];
        }
        const float nd = sqrt(zc_tg_reduce_add(nd_p, red, tid, ntg));
        const float nr = sqrt(zc_tg_reduce_add(nr_p, red, tid, ntg));
        const float cap = args.zl_tr * nr;
        const float s = (nd > cap && nd > 0.0f) ? (cap / nd) : 1.0f;
        for (uint j = tid; j < d; j += ntg) {
            float a = 0.0f;
            device const half *ur = hU + (uint64_t)j * zk;
            for (uint c = 0; c < zk; c++) a += pvS[c] * (float)ur[c];
            rt[j] += s * a;
        }
    }
}
