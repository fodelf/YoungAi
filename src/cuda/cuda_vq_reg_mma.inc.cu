/* cuda_vq_reg_mma.inc.cu — 专家张量核(前向)的"分段预取 + 寄存器直解"形态 vqs(2026-10-03), 替掉 cuda_vq_prefill_mma.inc.cu 的 vqm_kernel。
 * ★必须在 cuda_vq_prefill_mma.inc.cu 之后 include★(用它的 vqp_item / vqm_e4m3x2_to_bf16x2 / g_vqm 暂存), 被它的 vqm_run_impl 调。
 *
 * 【为什么】10-03 训练逐核表(batch 4, maxlen 880): 专家核占 GPU 时间 52.5%。微基准(gguf-tools/bench/v41_vq_train_bench.cu,
 * 真载荷 + 训练包真路由)量出 vqm_kernel 每轮 ~0.9 µs ≈ 一次 DRAM 延迟 —— 它只预取一轮(在途字节太少), 卡在延迟上;
 * 每解一个码字还要在 shared 里写一遍 A 瓦片再 ldmatrix 读回来。
 * 这一版: 一个 block(16 warp)管 (工作项, 256 行) × 全部 K;
 *   ① 位流与激活按 RS 轮(一轮 = 64 列)一段, cp.async 先搬进 shared, 两段在途, 一段一个屏障;
 *      每行的位流另按 384 B(3 条整线)一块提前两块 bulk 预取进 L2 —— 段内 cp.async 只读 L2, DRAM 看到的是整线连续请求
 *      (09-18 量过: 每条 load 读不满 128 B 整线只有 142 GB/s; 冷专家实测正卡在 ~130 GB/s, 加这一刀 8.6 → 6.2 ms);
 *   ② lane (g = lane/4, q = lane%4) 解单元内第 16w+g 与 16w+8+g 行的码字, 查完表(E4M3 码本, 8 B/词)现转 bf16 就是 mma 的
 *      A 片段(权重当 A、16 行; 激活当 B、8 个 token 一片), 不写 A/B 瓦片、不用 ldmatrix。
 *      k16 组内逻辑 k = 2q+{0,1} ↔ 物理列 16kk+4q+{0,1}, 2q+8+{0,1} ↔ 16kk+4q+{2,3}
 *      ⇒ 本 lane 每个 k16 要码字 2kk+q/2 的元素 4(q&1)..+3(半个码字, 一条 LDS.32); 激活同一组列 = 一条 8 B 读。
 * 微基准(ms/层, 前向三发, 12 位层 / 13 位层): 训练包(880 token)18.6 → 11.6 / 21.9 → 14.6; 预填块(2048 token)31.8 → 21.3 / 37.2 → 27.2。
 *
 * 【★逐位同★】mma 的 k16 组仍是同一组 16 列, 只换组内谁拿哪一列; 每个 k16 从零起算再 FADD 回累加器的次序、行增益与 bf16r 的出口
 * 一个没变 —— 微基准 g32/h16/H_u/ys 与 vqm_kernel 逐字节比, 12/13 位层 × 两种路由全同(张量核组内求和与元素次序无关)。
 * E4M3 → bf16 走硬件 cvt 到 f16 再经 f32 打包(vqs_e4m3x2_bf16x2), 与 vqm_e4m3x2_to_bf16x2 在全部非 NaN 输入上逐位同(码本无 NaN)。
 *
 * 【出错会怎样】激活段里 token t 的 16 B 块 c 存在 (c ^ 2(t&3)), 写入(issue)与取址(cofs)必须同一个式子; 位流段每行 12·RS 字节,
 * 读的是第 r 轮的 3 个字。写错不报错, 只是 mma 吃到别的列 —— 表现是 PPL 爆到几万; 门 = 微基准逐位门 + 引擎温 0 输出 cmp。 */
#include <type_traits>
#define VQS_THREADS 512u
#define VQS_BM      256u   /* 一个工作单元的行数: 16 warp × 16 行(M 必须是它的倍数: 2304 = 9·256, 5120 = 20·256) */
#define VQS_NTM     32u    /* 一个工作项最多几个 token(4 个 n8 片); 热专家按它切项, 微基准里 64 / 128 都不比它快 */

/* 一行这一轮的第 kk 个 k16 要的码字: 轮内码字号 c = 2kk + h(h = q/2), 位偏移 12c; 本轮 96 位在 w0..w2 */
__device__ __forceinline__ static uint32_t vqs_code(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t kk, uint32_t h) {
    if (kk == 0u) return (w0 >> (12u * h)) & 0xFFFu;
    if (kk == 1u) return (h ? (w1 >> 4) : __funnelshift_r(w0, w1, 24u)) & 0xFFFu;
    if (kk == 2u) return __funnelshift_r(w1, w2, 16u + 12u * h) & 0xFFFu;
    return (w2 >> (8u + 12u * h)) & 0xFFFu;
}
template <uint32_t N>
__device__ __forceinline__ static void vqs_cp(uint8_t *dst, const void *src) {   /* N = 16 走 .cg(只进 L2), 8/4 走 .ca */
    if (N == 16u) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
    else asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "n"(N) : "memory");
}
__device__ __forceinline__ static void vqs_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ static void vqs_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ static void vqs_pf_l2(const void *p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ static uint32_t vqs_e4m3x2_bf16x2(uint32_t two) {
    uint32_t hh, o;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hh) : "h"((unsigned short)(two & 0xffffu)));
    const float lo = __half2float(__ushort_as_half((unsigned short)(hh & 0xffffu))), hi = __half2float(__ushort_as_half((unsigned short)(hh >> 16)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o) : "f"(hi), "f"(lo));
    return o;
}
/* D = A·B(C 恒 +0, 输出单独一组寄存器)。不用 vqm_mma(t, …) 配 t = {0}: 那是 "+f" 读写约束, ptxas 把 D 排进 A 那组寄存器,
 * A 片段在多片 token 间复用时每条 HMMA 前要 4 条 MOV 重拼 A(10-03 SASS: 一段 32 条 HMMA 配 128 条 MOV)。数值同(C = +0)。 */
__device__ __forceinline__ static void vqs_mma0(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
/* shared 字节: 码本(E4M3 nc×8) + S 段 × (位流 256 行 × 12·RS B + 位平面 256 × 4 B + 激活 NTM × 128·RS B) */
static constexpr uint32_t vqs_smem(uint32_t cbb, int ext, uint32_t rs, uint32_t s) {
    return cbb + s * (VQS_BM * 12u * rs + (ext ? VQS_BM * 4u : 0u) + VQS_NTM * 128u * rs);
}

/* MODE 0 = gate: g32 = bf16(W1·x·g) | 1 = up: 读 g32 做 clamp+SwiGLU 写 h16(ys 非空 = 顺手写 H_u) | 2 = down: ys = bf16(W2·h·g)。
 * 参数与 vqm_kernel 一字不差(调用方 vqm_run_impl 同一套工作项/偏移/暂存), 只是工作项按 VQS_NTM 切。 */
template <int EXT, int MODE, uint32_t RS, uint32_t S>
__global__ __launch_bounds__(VQS_THREADS, 1) static void vqs_kernel(
        float *g32, uint16_t *h16, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
        const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp, uint32_t cb_bytes, const float *gr) {
    constexpr uint32_t NTN = VQS_NTM / 8u, RB = 12u * RS, CP = RS == 4u ? 16u : 8u;   /* 每行每段 RB 字节, 拆 3 块 CP 字节搬 */
    constexpr uint32_t BSB = VQS_BM * RB, EXB = EXT ? VQS_BM * 4u : 0u, ATB = VQS_NTM * 128u * RS, STB = BSB + EXB + ATB;
    constexpr uint32_t PCH = 384u, PD = 2u, SPC = PCH / RB;   /* L2 预取: 每行 384 B 一块, 提前 PD 块; SPC = 一块够几段 */
    static_assert(RS == 2u || RS == 4u, "一段 2 或 4 轮");
    extern __shared__ __align__(16) uint8_t vqssh[];
    uint8_t *cbs = vqssh, *ring = vqssh + cb_bytes;
    const int which = MODE;   /* 载荷槽: 0 w1(gate) / 1 w3(up) / 2 w2(down) */
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, q = lane & 3u, h = q >> 1, hf = q & 1u;
    {   /* v3 码本一层一本(三矩阵、全部专家共用), 原样 E4M3 进 shared */
        const v41_vq_mat m0 = v41_vq_open<1>(blob, items[0].e, which, M, K, NULL);
        if (!m0.ok || m0.nc * 8u != cb_bytes) return;   /* 整个 block 同一判断, 不会有人卡在后面的 barrier 上 */
        v41_vq_cb_to_shared(cbs, m0.cb, cb_bytes);
    }
    __syncthreads();
    const uint32_t ntile = M / VQS_BM, nwork = nitems * ntile, nst = K / (64u * RS);
    const uint32_t la = warp * 16u + g, lb = la + 8u;
    /* 本线程搬位流的两块(k = tid, tid+512, 只有 < 768 的才有): 行 k/3 的第 k%3 块 */
    const uint32_t k1 = tid + VQS_THREADS, rw0 = tid / 3u, pt0 = tid - rw0 * 3u, rw1 = k1 / 3u, pt1 = k1 - rw1 * 3u;
    const uint32_t cofs[4] = { ((0u + h) ^ (2u * (g & 3u))) << 4, ((2u + h) ^ (2u * (g & 3u))) << 4,
                               ((4u + h) ^ (2u * (g & 3u))) << 4, ((6u + h) ^ (2u * (g & 3u))) << 4 };
    for (uint32_t w = blockIdx.x; w < nwork; w += gridDim.x) {
        const vqp_item it = items[w / ntile];
        const uint32_t r0 = (w % ntile) * VQS_BM, nt = (uint32_t)it.nt, base = off[it.e] + (uint32_t)it.t0;
        const v41_vq_mat m = v41_vq_open<1>(blob, it.e, which, M, K, (MODE == 2 && gr) ? gr + (size_t)it.e * M : NULL);
        if (!m.ok) {   /* 主机侧已按槽表查过, 走到这里 = 载荷坏了; down 写 0 不给 reduce 留脏值(与 vqm_kernel 同) */
            if (MODE == 2)
                for (uint32_t i = tid; i < VQS_BM * nt; i += VQS_THREADS) ys[(uint64_t)(base + i / VQS_BM) * M + r0 + i % VQS_BM] = 0.f;
            continue;
        }
        const uint32_t mrow = m.nidx_row * 12u / 8u, erow = (m.nidx_row + 7u) >> 3, nach = nt * 8u * RS;
        const uint8_t *prow = m.ix + (size_t)(r0 + (tid & 255u)) * mrow;
        auto pf_chunk = [&](uint32_t c) {
            if (tid < VQS_BM && c * PCH < mrow) vqs_pf_l2(prow + c * PCH, mrow - c * PCH < PCH ? mrow - c * PCH : PCH);
        };
        #pragma unroll
        for (uint32_t c = 0; c < PD; c++) pf_chunk(c);
        if (EXT && tid == 0) vqs_pf_l2(m.ex + (size_t)r0 * erow, (VQS_BM * erow) & ~15u);   /* 位平面整段只有 9~20 KB, 一条拿完 */
        const uint8_t *bsrc0 = m.ix + (size_t)(r0 + rw0) * mrow + CP * pt0, *bsrc1 = m.ix + (size_t)(r0 + (k1 < 768u ? rw1 : 0u)) * mrow + CP * pt1;
        const uint8_t *esrc = EXT ? m.ex + (size_t)(r0 + (tid & 255u)) * erow : NULL;
        const uint16_t *xa = act + (uint64_t)base * K;
        auto issue = [&](uint32_t s) {   /* 第 s 段(轮 RS·s ..)→ 槽 s % S */
            uint8_t *st = ring + (s % S) * STB;
            vqs_cp<CP>(st + rw0 * RB + CP * pt0, bsrc0 + RB * s);
            if (k1 < 768u) vqs_cp<CP>(st + rw1 * RB + CP * pt1, bsrc1 + RB * s);
            if (EXT && tid < VQS_BM) vqs_cp<4>(st + BSB + tid * 4u, esrc + 4u * ((RS * s) >> 2));   /* 位平面按 4 轮一字搬, RS=2 时相邻两段搬同一字 */
            uint8_t *ad = st + BSB + EXB;
            for (uint32_t k = tid; k < nach; k += VQS_THREADS) {
                const uint32_t t = k / (8u * RS), rem = k - t * 8u * RS, r = rem >> 3, c = rem & 7u;
                vqs_cp<16>(ad + (r * VQS_NTM + t) * 128u + ((c ^ (2u * (t & 3u))) << 4), xa + (uint64_t)t * K + 64u * (RS * s + r) + 8u * c);
            }
        };
        /* 序幕: 第 0..S-2 段各一组。之后每段提交一组 ⇒ 第 s 段在第 s 组, wait_group(S-2) 刚好等到它 */
        #pragma unroll
        for (uint32_t s = 0; s + 1u < S; s++) { if (s < nst) issue(s); vqs_commit(); }
        const uint32_t nnt = (nt + 7u) >> 3;
        float acc[NTN][4];
        #pragma unroll
        for (uint32_t j = 0; j < NTN; j++) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
        /* 段循环按本项的 n8 片数 NNT 实例化(编译期常量): 片循环没有逐片分支 */
        auto run = [&](auto nntc) {
            constexpr uint32_t NNT = decltype(nntc)::value;
            for (uint32_t s = 0; s < nst; s++) {
                vqs_wait<(int)S - 2>();
                __syncthreads();   /* 第 s 段全到了; 也保证大家都用完了第 s-1 段的槽 ⇒ 下面可以往里搬第 s+S-1 段 */
                if (s + S - 1u < nst) issue(s + S - 1u);
                vqs_commit();
                if (s % SPC == 0u) pf_chunk(s / SPC + PD);
                const uint8_t *st = ring + (s % S) * STB;
                uint32_t wa[3 * RS], wb[3 * RS];
                if (RS == 4u) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 3u; k++) {
                        const uint4 u = *(const uint4 *)(st + la * RB + 16u * k), v = *(const uint4 *)(st + lb * RB + 16u * k);
                        wa[4 * k] = u.x; wa[4 * k + 1] = u.y; wa[4 * k + 2] = u.z; wa[4 * k + 3] = u.w;
                        wb[4 * k] = v.x; wb[4 * k + 1] = v.y; wb[4 * k + 2] = v.z; wb[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (uint32_t k = 0; k < 3u; k++) {
                        const uint2 u = *(const uint2 *)(st + la * RB + 8u * k), v = *(const uint2 *)(st + lb * RB + 8u * k);
                        wa[2 * k] = u.x; wa[2 * k + 1] = u.y; wb[2 * k] = v.x; wb[2 * k + 1] = v.y;
                    }
                }
                uint32_t xe = 0u, ye = 0u;
                if (EXT) { xe = *(const uint32_t *)(st + BSB + la * 4u); ye = *(const uint32_t *)(st + BSB + lb * 4u); }
                const uint8_t *as = st + BSB + EXB + 8u * hf + g * 128u;
                #pragma unroll
                for (uint32_t r = 0; r < RS; r++) {
                    const uint32_t eb = ((RS * s + r) & 3u) * 8u;   /* 本轮在位平面字里的字节 */
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; kk++) {
                        uint32_t va = vqs_code(wa[3 * r], wa[3 * r + 1], wa[3 * r + 2], kk, h), vb = vqs_code(wb[3 * r], wb[3 * r + 1], wb[3 * r + 2], kk, h);
                        if (EXT) { const uint32_t c = eb + 2u * kk + h; va |= ((xe >> c) & 1u) << 12; vb |= ((ye >> c) & 1u) << 12; }
                        const uint32_t ea = *(const uint32_t *)(cbs + (size_t)va * 8u + hf * 4u), ebb = *(const uint32_t *)(cbs + (size_t)vb * 8u + hf * 4u);
                        const uint32_t a[4] = { vqs_e4m3x2_bf16x2(ea), vqs_e4m3x2_bf16x2(ebb), vqs_e4m3x2_bf16x2(ea >> 16), vqs_e4m3x2_bf16x2(ebb >> 16) };
                        const uint8_t *ap = as + r * VQS_NTM * 128u + cofs[kk];
                        #pragma unroll
                        for (uint32_t j = 0; j < NNT; j++) {
                            const uint2 x = *(const uint2 *)(ap + j * 8u * 128u);
                            /* ★每个 k16 从零起算, 结果用普通 FADD 加回累加器★(同 vqm_kernel: 张量核内部长 K 累加对齐时截断, 直接连乘单向丢低位) */
                            float t[4];
                            vqs_mma0(t, a, x.x, x.y);
                            #pragma unroll
                            for (int c = 0; c < 4; c++) acc[j][c] += t[c];
                        }
                    }
                }
            }
        };
#define VQS_NNT(n) std::integral_constant<uint32_t, ((n) < NTN ? (n) : NTN)>{}
        switch (nnt) {
        case 1: run(VQS_NNT(1)); break;
        case 2: run(VQS_NNT(2)); break;
        case 3: run(VQS_NNT(3)); break;
        default: run(VQS_NNT(4)); break;
        }
#undef VQS_NNT
        __syncthreads();   /* 下一个单元的序幕要覆盖这几个槽 */
        /* 出口: c0,c1 = 行 16w+g、token 8j+2q+{0,1}; c2,c3 = 行 +8。舍入点与 vqm_kernel 同(求和 → 乘行增益 → bf16r) */
        const uint32_t ra = r0 + la, rb = r0 + lb;
        float gna, gnb;
        { __half gh; memcpy(&gh, m.gr + (size_t)ra * 2u, 2); gna = __half2float(gh) * (m.gov ? m.gov[ra] : 1.0f);
          memcpy(&gh, m.gr + (size_t)rb * 2u, 2); gnb = __half2float(gh) * (m.gov ? m.gov[rb] : 1.0f); }
        #pragma unroll
        for (uint32_t j = 0; j < NTN; j++) {
            if (j >= nnt) break;
            #pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t t = 8u * j + 2u * q + e;
                if (t >= nt) continue;
                const uint64_t o = (uint64_t)(base + t) * M;
                const float va = v41_bf16r(acc[j][e] * gna), vb = v41_bf16r(acc[j][2u + e] * gnb);
                if (MODE == 0) { g32[o + ra] = va; g32[o + rb] = vb; }
                else if (MODE == 1) { h16[o + ra] = v41_vq_swiglu(g32[o + ra], va, clamp); h16[o + rb] = v41_vq_swiglu(g32[o + rb], vb, clamp);
                                      if (ys) { ys[o + ra] = va; ys[o + rb] = vb; } }
                else { ys[o + ra] = va; ys[o + rb] = vb; }
            }
        }
    }
}

static uint32_t vqs_item_tokens(void) { return VQS_NTM; }
/* ★vqs / vqst 的前提: 位流 16 B 对齐 + 设备内存★(cp.async 16 B 与 bulk 预取进 L2 都要)。只有启动缓存的设备副本(cuda_vq_align 把载荷挪到 128 B 对齐)满足;
 * 封顶装载(--weight-cache-mb)装不下的层走 cudaHostRegister 直接映射原始 GGUF 字节, 载荷只 8 B 对齐 —— 10-03 实撞: wt2 门(封顶 88000)的 L31 前向失败,
 * 而全驻留的训练/ab 档看不见。不满足就照旧走瓦片核 vqm_kernel(它只按 4 B 读位流)。按 (层, blob 指针) 记一次, 换了指针再查。 */
static struct { const uint8_t *blob; int ok; } g_vqs_fit[64];
static int vqs_blob_fits(uint32_t layer, const uint8_t *blob, uint32_t n_total, uint32_t IN, uint32_t MID, uint32_t OUT) {
    if (layer >= 64u) return 0;
    if (g_vqs_fit[layer].blob == blob) return g_vqs_fit[layer].ok;
    if (!g_vqp_hdr[layer] && !vqp_hdr_build(layer, blob, n_total, IN, MID, OUT, 3u)) return 0;
    int ok = ((uintptr_t)blob & 15u) == 0u;
    cudaPointerAttributes at;
    if (ok && (cudaPointerGetAttributes(&at, blob) != cudaSuccess || at.type != cudaMemoryTypeDevice)) { (void)cudaGetLastError(); ok = 0; }
    for (uint32_t k = 0; ok && k < n_total * 3u; k++) {   /* 位流起点 = 载荷 + 32 + 行增益 rows×2 */
        const uint64_t off = g_vqp_hdr[layer][k].off;
        if (off && ((off + 32u + (uint64_t)((k % 3u) == 2u ? OUT : MID) * 2u) & 15u)) ok = 0;
    }
    if (!ok) { static int said; if (!said++) fprintf(stderr, "ds4: [vq-prefill] L%u 专家 blob 不是对齐的设备副本(映射/平拷), 这类层走瓦片核\n", layer); }
    g_vqs_fit[layer].blob = blob; g_vqs_fit[layer].ok = ok;
    return ok;
}
/* vqs 认不认这个形状: M(gate/up 的 MID, down 的 OUT)是 256 的倍数, K 是一段(64·RS 列)的倍数 */
static bool vqs_shape_ok(uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nbit) {
    const uint32_t seg = 64u * (nbit == 13u ? 2u : 4u);
    return (nbit == 12u || nbit == 13u) && MID % VQS_BM == 0u && OUT % VQS_BM == 0u && IN % seg == 0u && MID % seg == 0u;
}
/* 三发(gate → up → down)。13 位层码本 64 KB, shared 只够一段 2 轮; 12 位层码本 32 KB, 一段 4 轮(屏障少一半, 微基准快 7%) */
static int vqs_launch3(float *g32, uint16_t *h16, float *hu, float *ys, const uint8_t *blob, const vqp_item *items, uint32_t nitems,
                       const uint16_t *xs, const uint32_t *doff, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, uint32_t nbit,
                       float clamp, const float *gr, uint32_t layer_index) {
    static int occ[2][3];
    const uint32_t cbb = nc * 8u;
#define VQS_GO(E, RS_) do { \
        const uint32_t shb = vqs_smem(cbb, E, RS_, 2u); \
        int *oc = occ[E]; \
        if (!oc[0]) { \
            const bool ok = cudaFuncSetAttribute(vqs_kernel<E, 0, RS_, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) == cudaSuccess && \
                            cudaFuncSetAttribute(vqs_kernel<E, 1, RS_, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) == cudaSuccess && \
                            cudaFuncSetAttribute(vqs_kernel<E, 2, RS_, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb) == cudaSuccess && \
                            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oc[0], vqs_kernel<E, 0, RS_, 2>, (int)VQS_THREADS, shb) == cudaSuccess && oc[0] > 0; \
            if (!ok) { (void)cudaGetLastError(); oc[0] = -1; } \
        } \
        if (oc[0] < 0) { fprintf(stderr, "ds4: [vq-prefill] L%u 寄存器直解核开不出 %u KB shared\n", layer_index, shb >> 10); return 0; } \
        const uint32_t gcap = (uint32_t)g_vqm.nsm * (uint32_t)oc[0], ng = nitems * (MID / VQS_BM), nd = nitems * (OUT / VQS_BM); \
        vqs_kernel<E, 0, RS_, 2><<<ng < gcap ? ng : gcap, VQS_THREADS, shb, g_cur_stream>>>(g32, NULL, NULL, blob, items, nitems, xs, doff, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill reg gate")) return 0; \
        vqs_kernel<E, 1, RS_, 2><<<ng < gcap ? ng : gcap, VQS_THREADS, shb, g_cur_stream>>>(g32, h16, hu, blob, items, nitems, xs, doff, MID, IN, clamp, cbb, NULL); \
        if (!cuda_ok(cudaGetLastError(), "vq prefill reg up")) return 0; \
        vqs_kernel<E, 2, RS_, 2><<<nd < gcap ? nd : gcap, VQS_THREADS, shb, g_cur_stream>>>(NULL, NULL, ys, blob, items, nitems, h16, doff, OUT, MID, clamp, cbb, gr); \
        return cuda_ok(cudaGetLastError(), "vq prefill reg down"); \
    } while (0)
    if (nbit == 13u) VQS_GO(1, 2u);
    VQS_GO(0, 4u);
#undef VQS_GO
}
