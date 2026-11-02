/* cuda_bwd_vgrad.inc.cu — V 步(离散码字重选, 2026-11-02)的按组权重梯度: 选中专家的 gate/up/down 对去量化权重的梯度 dW = Gᵀ·X(含行增益),
 * 在 routed 专家反传里顺手累加进设备缓冲(跨多批累加), 之后 cuda_vq_vstep.inc.cu 按组对近邻码字算一阶 Δ 选码。
 *
 * 为什么要它: 专家 1.5 bit 的尾巴(fable5 11-02: 判决份 KLD>2 的 177 位、鹈鹕"喉"位 0.012)任何连续小参数都没动过 —— 行增益/路由偏置(②)、
 *   层码本 δ(10-29 码本牵引, 不迁)、旁路低秩(③ 六轮)全是 PV-Tuning 说的"P 步"; 2 bit 以下收益在"V 步": 每组重选码字。V 步要的就是每组 8 个权重的梯度。
 * 与 cuda_bwd_cbgrad(10-29, stash)同一个核骨架(一块 = 一个专家 × 64 行 × 128 列 = 16 个码字列, bf16 张量核 vqm_mma), 只把"按码字号归并"
 *   换成"按 (专家槽, 矩阵, 行, 列) 落地", 且不抽样(V 步要整份梯度)。
 * 布局: dw[slot][gate MID×IN | up MID×IN | down OUT×MID] f32, slot = slot_of[e](设备 int32 [n_expert], −1 = 不收)。值 = gain(e,r)·Σ_p G[p][r]·X[p][c]
 *   —— 已含行增益(含 ② 的覆盖), 所以 Δ(换码字) = <dw 组, C_new − C_cur>, 不用再乘增益。
 * 内存: V4.1 Flash 一个专家三矩阵 f32 ≈ 141 MB, 4 槽 ≈ 0.56 GB(训练态可用 ~6 GB)。
 * 出错会怎样: 形状不整除 64×128 ⇒ 返回 0 停车(核按整块写); dw == NULL 就是关, 反传路一行不多跑。 */
#define VG_RT 64u
#define VG_CT 128u
#define VG_TK 32u
#define VG_PAD 40u

static struct { ds4_gpu_tensor *dw; const ds4_gpu_tensor *slot_of; uint32_t nslot, IN, MID, OUT; uint32_t *counts, cnt_layers, cnt_experts; } g_vg;

int ds4_gpu_bwd_moe_vgrad(ds4_gpu_tensor *dw, const ds4_gpu_tensor *slot_of, uint32_t nslot, uint32_t IN, uint32_t MID, uint32_t OUT) {
    g_vg.dw = dw; g_vg.slot_of = slot_of; g_vg.nslot = nslot; g_vg.IN = IN; g_vg.MID = MID; g_vg.OUT = OUT;
    return 1;
}
/* 路由计数(主机 [n_layer][n_expert], NULL = 关): 反传每过一层把该层各专家本批配对数加进去 —— V 步挑"热门专家"用(收益 ∝ 路由份额, 10-21 拆账) */
int ds4_gpu_bwd_moe_count(uint32_t *counts, uint32_t n_layer, uint32_t n_expert) { g_vg.counts = counts; g_vg.cnt_layers = n_layer; g_vg.cnt_experts = n_expert; return 1; }

/* 槽内三矩阵的 float 偏移(与主机 pt_vstep 同式; 改一处另一处必改) */
__host__ __device__ static inline uint64_t vg_slot_floats(uint32_t IN, uint32_t MID, uint32_t OUT) { return 2ull * MID * IN + (uint64_t)OUT * MID; }
__host__ __device__ static inline uint64_t vg_which_off(uint32_t which, uint32_t IN, uint32_t MID) { return which == 0u ? 0ull : which == 1u ? (uint64_t)MID * IN : 2ull * MID * IN; }

/* XF = 1: X 按 token 存 f32 [n_tok][C], 配对行 p 读第 pair[p]/K 行(gate/up 的输入 x); 0: X 按配对存 bf16 [nv][C](down 的输入 A) */
template <int XF>
__global__ __launch_bounds__(128) static void vg_kernel(float *dw, const int32_t *slot_of, uint64_t slot_floats, uint64_t which_off,
        const uint8_t *blob, uint32_t which, uint32_t R, uint32_t C, const float *G, const void *X, const int32_t *pair, uint32_t K,
        const uint32_t *act, const uint32_t *aoff, const uint32_t *acnt, const float *gr_all, uint32_t OUTd, int *bad) {
    __shared__ __align__(16) __nv_bfloat16 gs[VG_RT * VG_PAD];
    __shared__ __align__(16) __nv_bfloat16 xs[VG_CT * VG_PAD];
    const uint32_t a = blockIdx.z, e = act[a], p0 = aoff[a], T = acnt[a];
    const int32_t slot = slot_of[e];
    if (slot < 0) return;   /* 整块同一个判断 */
    const uint32_t c0 = blockIdx.x * VG_CT, r0 = blockIdx.y * VG_RT;
    const v41_vq_mat mm = v41_vq_open<1>(blob, (int)e, (int)which, R, C, (which == 2u && gr_all) ? gr_all + (size_t)e * OUTd : NULL);
    if (!mm.ok) { if (threadIdx.x == 0) atomicExch(bad, 1); return; }
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, q = lane & 3u;
    float acc[16][4];
    #pragma unroll
    for (uint32_t nt = 0; nt < 16u; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }
    for (uint32_t t0 = 0; t0 < T; t0 += VG_TK) {
        const uint32_t tn = T - t0 < VG_TK ? T - t0 : VG_TK;
        __syncthreads();
        for (uint32_t i = tid; i < VG_TK * (VG_RT / 4u); i += blockDim.x) {
            const uint32_t t = i / (VG_RT / 4u), r4 = (i % (VG_RT / 4u)) * 4u;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < tn) v = *(const float4 *)(G + (uint64_t)(p0 + t0 + t) * R + r0 + r4);
            gs[(r4 + 0u) * VG_PAD + t] = __float2bfloat16(v.x); gs[(r4 + 1u) * VG_PAD + t] = __float2bfloat16(v.y);
            gs[(r4 + 2u) * VG_PAD + t] = __float2bfloat16(v.z); gs[(r4 + 3u) * VG_PAD + t] = __float2bfloat16(v.w);
        }
        if (XF) {
            for (uint32_t i = tid; i < VG_TK * (VG_CT / 4u); i += blockDim.x) {
                const uint32_t t = i / (VG_CT / 4u), c4 = (i % (VG_CT / 4u)) * 4u;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (t < tn) v = *(const float4 *)((const float *)X + (uint64_t)(pair[p0 + t0 + t] / (int32_t)K) * C + c0 + c4);
                xs[(c4 + 0u) * VG_PAD + t] = __float2bfloat16(v.x); xs[(c4 + 1u) * VG_PAD + t] = __float2bfloat16(v.y);
                xs[(c4 + 2u) * VG_PAD + t] = __float2bfloat16(v.z); xs[(c4 + 3u) * VG_PAD + t] = __float2bfloat16(v.w);
            }
        } else {
            for (uint32_t i = tid; i < VG_TK * (VG_CT / 8u); i += blockDim.x) {
                const uint32_t t = i / (VG_CT / 8u), c8 = (i % (VG_CT / 8u)) * 8u;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (t < tn) v = *(const uint4 *)((const __nv_bfloat16 *)X + (uint64_t)(p0 + t0 + t) * C + c0 + c8);
                const __nv_bfloat16 *h = (const __nv_bfloat16 *)&v;
                #pragma unroll
                for (uint32_t k = 0; k < 8u; k++) xs[(c8 + k) * VG_PAD + t] = h[k];
            }
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < VG_TK; kk += 16u) {
            const __nv_bfloat16 *ga = gs + (warp * 16u + g) * VG_PAD + kk + 2u * q;
            uint32_t af[4];
            af[0] = *(const uint32_t *)ga; af[1] = *(const uint32_t *)(ga + 8u * VG_PAD);
            af[2] = *(const uint32_t *)(ga + 8u); af[3] = *(const uint32_t *)(ga + 8u * VG_PAD + 8u);
            #pragma unroll
            for (uint32_t nt = 0; nt < 16u; nt++) {
                const __nv_bfloat16 *xb = xs + (nt * 8u + g) * VG_PAD + kk + 2u * q;
                vqm_mma(acc[nt], af, *(const uint32_t *)xb, *(const uint32_t *)(xb + 8u));
            }
        }
    }
    /* C 片: c0,c1 = 行 g 维 2q,2q+1; c2,c3 = 行 g+8 同两维。q 偶数的 lane 收齐行 g 的维 2q..2q+3, q 奇数的收齐行 g+8 的维 2q-2..2q+1 */
    const bool ev = (q & 1u) == 0u;
    const uint32_t row = r0 + warp * 16u + g + (ev ? 0u : 8u), db = ev ? 2u * q : 2u * q - 2u;
    const float gn = vqb_gain(mm, row);
    float *base = dw + (uint64_t)slot * slot_floats + which_off + (uint64_t)row * C;
    #pragma unroll
    for (uint32_t nt = 0; nt < 16u; nt++) {
        const float s0 = ev ? acc[nt][2] : acc[nt][0], s1 = ev ? acc[nt][3] : acc[nt][1];
        const float u0 = __shfl_xor_sync(0xffffffffu, s0, 1), u1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        const float4 v = ev ? make_float4(acc[nt][0] * gn, acc[nt][1] * gn, u0 * gn, u1 * gn)
                            : make_float4(u0 * gn, u1 * gn, acc[nt][2] * gn, acc[nt][3] * gn);
        atomicAdd((float4 *)(base + c0 + nt * 8u + db), v);   /* 跨批累加; 同一发里每个 (槽, 行, 列) 只有一个 lane 写, 原子只为跨发安全 */
    }
}

static int vg_launch(uint32_t which, uint32_t R, uint32_t C, const float *G, const void *X, int xf, const int32_t *pair, uint32_t K,
                     const uint32_t *act, const uint32_t *aoff, const uint32_t *acnt, uint32_t nact, const uint8_t *blob,
                     const float *gr_all, uint32_t OUTd, int *bad) {
    if (R % VG_RT || C % VG_CT) { fprintf(stderr, "ds4: [V步·梯度] 矩阵 %u 形状 %u×%u 不整除 %u×%u\n", which, R, C, VG_RT, VG_CT); return 0; }
    if (!nact) return 1;
    const uint64_t sf = vg_slot_floats(g_vg.IN, g_vg.MID, g_vg.OUT), wo = vg_which_off(which, g_vg.IN, g_vg.MID);
    const dim3 grid(C / VG_CT, R / VG_RT, nact);
    float *dw = (float *)g_vg.dw->ptr; const int32_t *so = (const int32_t *)g_vg.slot_of->ptr;
    if (xf) vg_kernel<1><<<grid, 128, 0, g_cur_stream>>>(dw, so, sf, wo, blob, which, R, C, G, X, pair, K, act, aoff, acnt, gr_all, OUTd, bad);
    else    vg_kernel<0><<<grid, 128, 0, g_cur_stream>>>(dw, so, sf, wo, blob, which, R, C, G, X, pair, K, act, aoff, acnt, gr_all, OUTd, bad);
    return cuda_ok(cudaGetLastError(), "bwd vgrad");
}
