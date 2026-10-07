/* v41_vq_persist_mma.cuh — 解码路专家常驻核的张量核形态(2026-10-07), 被 v41_vq_persist_n_bench.cu 包含(用它的 mat_t/MAT/groups/SEG/常量)。
 *
 * 【为什么】ncu(10-07, n=4 真路由重合度): gu/dn 两个常驻核每 32 个码字 86 条指令, 发射槽 72~73% 忙、访存只到峰值一半 ——
 * 专家核卡的是指令数, 不是带宽(09-23 "不是被指令条数卡住"是 n=1 M=1 的结论, 验证批 M=2 每轮多付一整套 bf16 拆包 + FMA)。
 * 【怎么改】与预填 vqs 核(src/cuda/cuda_vq_reg_mma.inc.cu)同一套数值: 权重当 A(16 行 × k16)、激活当 B(8 个 token 一片),
 *   每个 k16 从零起算 mma 再 FADD 回累加器, 行增益 + bf16r 出口 —— 解码路与预填路**逐位同**成为可能(以前是两套累加序)。
 *   lane (g = lane/4, q = lane%4) 解第 r0+g 与 r0+g+8 行的码字: k16 组内逻辑 k = 2q+{0,1} ↔ 物理列 16kk+4q+{0,1},
 *   2q+8+{0,1} ↔ 16kk+4q+{2,3} ⇒ 本 lane 每个 k16 要码字 2kk+q/2 的元素 4(q&1)..+3(半个码字), 激活同一组列 = 一条 8 B 读。
 *   12 位层: 码本在 shared 里存成 bf16(4096 词 × 16 B = 64 KB), 半个码字 = 一条 LDS.64, 零转换指令;
 *   13 位层: 码本 8192 词只装得下 E4M3(64 KB), 半个码字 = 一条 LDS.32 + 两次 cvt 链(与 vqs 同式); 第 13 位走位平面(每块 32 码字 = 1 字)。
 *   位流按块(32 码字 = 48 B)读: 4 个 lane 各 3 个字(LDG.32 × 3), 第 s 步(8 码字)的 3 个字正好是 lane s 的 3 个寄存器 ⇒ 3 条 shfl 广播,
 *   不再有 sel3/funnelshift 那一套; 下一块在算本块前已发(寄存器提前一块), 再往后两块用 prefetch.global.L2 先送进 L2。
 *   工作划分不变: 每 warp 一段连续 (组, 行), 行数相等 ⇒ 无尾巴; 段内按 16 行一瓦片, 末瓦片不足 16 行的 lane 读 clamp 到末行、不写。
 * 【判据】ref_mma(同一 mma、最直白的逐 k16 实现)逐位同 ⇒ 块/shfl/预取/分段没错; 与 V0(f32 FMA 链)比最大相对差 ~1e-6 档 ⇒ 数学没错。 */
#define TM_MAXTOK 8u   /* 一组最多 8 个 token(一片 n8) */

__device__ __forceinline__ static uint32_t tm_e4m3x2_bf16x2(uint32_t two) {   /* = vqs_e4m3x2_bf16x2(非 NaN 输入精确) */
    uint32_t hh, o;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hh) : "h"((unsigned short)(two & 0xffffu)));
    const float lo = __half2float(__ushort_as_half((unsigned short)(hh & 0xffffu))), hi = __half2float(__ushort_as_half((unsigned short)(hh >> 16)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o) : "f"(hi), "f"(lo));
    return o;
}
__device__ __forceinline__ static void tm_mma0(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {   /* = vqs_mma0: D = A·B, C = +0 */
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
__device__ __forceinline__ static uint32_t tm_code(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t kk, uint32_t h) {   /* = vqs_code */
    if (kk == 0u) return (w0 >> (12u * h)) & 0xFFFu;
    if (kk == 1u) return (h ? (w1 >> 4) : __funnelshift_r(w0, w1, 24u)) & 0xFFFu;
    if (kk == 2u) return __funnelshift_r(w1, w2, 16u + 12u * h) & 0xFFFu;
    return (w2 >> (8u + 12u * h)) & 0xFFFu;
}
__device__ __forceinline__ static void tm_pf(const void *p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }

/* 码本进 shared: 13 位层原样 E4M3(8 B/词), 12 位层转 bf16(16 B/词); 两档都是 64 KB */
template <int EXT>
__device__ __forceinline__ static void tm_stage_cb(uint8_t *sh, const uint8_t *cb, uint32_t nwords) {
    if (EXT) { for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) ((uint2 *)sh)[i] = ((const uint2 *)cb)[i]; }
    else for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
        const uint2 w = ((const uint2 *)cb)[i];
        uint4 o; o.x = tm_e4m3x2_bf16x2(w.x); o.y = tm_e4m3x2_bf16x2(w.x >> 16); o.z = tm_e4m3x2_bf16x2(w.y); o.w = tm_e4m3x2_bf16x2(w.y >> 16);
        ((uint4 *)sh)[i] = o;
    }
}
/* 整个码字 → 4 个 bf16x2(元素 (0,1) (2,3) (4,5) (6,7)): 12 位层一条 LDS.128; 13 位层一条 LDS.64 + 4 次 cvt 链 */
template <int EXT>
__device__ __forceinline__ static uint4 tm_word(const uint8_t *cbs, uint32_t v) {
    if (EXT) { const uint2 e = *(const uint2 *)(cbs + (size_t)v * 8u); uint4 o;
               o.x = tm_e4m3x2_bf16x2(e.x); o.y = tm_e4m3x2_bf16x2(e.x >> 16); o.z = tm_e4m3x2_bf16x2(e.y); o.w = tm_e4m3x2_bf16x2(e.y >> 16); return o; }
    return *(const uint4 *)(cbs + (size_t)v * 16u);
}
/* 第二版(10-07 下午): 第一版 ncu 说卡在 MIO(码本 LDS.64 随机地址 4.6 波前/条 + 每 k16 两条)和 L1(位流流式读把激活挤出 L1, 命中 62%)。
 * ① 一 lane 一次取整个码字, 一个码字喂两个 k16: 一步 8 码字里 lane q 取码字 4p+q(p = 0,1), 元素 (0..3) 当第 2p 组、(4..7) 当第 2p+1 组
 *    (k16 组内逻辑 k ↔ 物理列的映射两组各自成立; 激活同样 16 B 一条读) ⇒ LDS 与码字提取都减半;
 * ② 位流用 __ldcg(只进 L2, 不占 L1): lane q<3 各一条 16 B 读, 块(48 B)= 3 lane; 第 s 步的 3 个字 = (lane, 分量)编译期常量 ⇒ 3 条 shfl;
 * ③ 不再寄存器提前一块(省 10 个寄存器), 只靠 prefetch.global.L2 提前三块 + 32 个 warp 互相盖延迟。
 * ★分组变了 ⇒ 与 vqs(预填)不再逐位同★: vqs 一组 = 连续 16 列, 这里一组 = 4 个码字各取半; 解码路自己(n=1 与验证批)仍逐位同。 */
/* 第三版(10-07 傍晚): 第二版 ncu 说块首位流读的延迟(带载 ~1 µs)全暴露(long_scoreboard 10.5/指令), 1024 线程下 64 个寄存器放不下提前读。
 * 改 512 线程/块(每线程 128 个寄存器): 位流在寄存器里提前两块(cur/n1/n2 轮转, 每块 8 个寄存器), 块首不再等。 */
#define TM_NW 16u   /* 每块 warp 数 */
/* 第四版(10-07 晚): 第三版 ncu 剩的最大项是 long_scoreboard 7.3/指令 —— 码本查表/激活读的结果被紧随的 mma 立刻用, 16 warp 每调度器只 4 个,
 * 盖不住 L1 命中的 ~40 周期。改手工流水: 一块的 4 步 24 个字先 shfl 齐(寄存器够), 8 个码字组按序走, 每组先发**下一组**的两条查表 + 激活读, 再做本组的 mma。
 * HALF=1: 瓦片 ≤ 8 行(尾瓦片), 下半行(lb)的读/shfl/提取/查表全跳过, A 片段下半填 0 —— n=1 时每 warp 18 行 = 16 + 2, 原来第二瓦片 7/8 的解码是重复算末行。 */
template <int EXT, uint32_t NI, int HALF>
__device__ __forceinline__ static void tm_tile(const mat_t &m, const uint8_t *exm, uint32_t r0, uint32_t n, const uint32_t *xrow,
                                               const uint8_t *cbs, float *acc) {
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, g = lane >> 2, qb = lane & ~3u;
    constexpr uint32_t NCH = NI / 32u, EROW = NI / 8u;
    const uint32_t ra = r0 + (g < n ? g : n - 1u), rb = r0 + (g + 8u < n ? g + 8u : n - 1u);   /* 越界行 clamp 到末行(只读不写) */
    const uint8_t *rowa = m.ix + (size_t)ra * m.rowb, *rowb = m.ix + (size_t)rb * m.rowb;
    const uint8_t *exa = EXT ? exm + (size_t)ra * EROW : NULL, *exb = EXT ? exm + (size_t)rb * EROW : NULL;
    const uint32_t sh0 = (12u * q) & 31u, sh1 = (48u + 12u * q) & 31u;
    const bool q3 = q == 3u, q2 = q >= 2u;   /* p=0: q<3 ⇒ 字 0/1, q=3 ⇒ 字 1/2;  p=1: q<2 ⇒ 字 1/2, q≥2 ⇒ 字 2/(无) */
    #define TM_PF(C) do { if ((C) < NCH) { if (q == 0u) tm_pf(rowa + 48u * (C)); else if (!HALF && q == 1u) tm_pf(rowb + 48u * (C)); } } while (0)
    #define TM_LD(C, PA, PB, XA, XB) do { \
        if ((C) < NCH && q < 3u) { PA = __ldcg((const uint4 *)(rowa + 48u * (C) + 16u * q)); if (!HALF) PB = __ldcg((const uint4 *)(rowb + 48u * (C) + 16u * q)); } \
        if (EXT && (C) < NCH) { XA = __ldcg((const uint32_t *)(exa + 4u * (C))); if (!HALF) XB = __ldcg((const uint32_t *)(exb + 4u * (C))); } } while (0)
    /* 第 s 步的 3 个字: (lane 偏移, 分量) 编译期常量 */
    #define TM_SHF(S, P, W0, W1, W2) do { \
        if ((S) == 0u)      { W0 = __shfl_sync(0xffffffffu, (P).x, qb);      W1 = __shfl_sync(0xffffffffu, (P).y, qb);      W2 = __shfl_sync(0xffffffffu, (P).z, qb); } \
        else if ((S) == 1u) { W0 = __shfl_sync(0xffffffffu, (P).w, qb);      W1 = __shfl_sync(0xffffffffu, (P).x, qb + 1u); W2 = __shfl_sync(0xffffffffu, (P).y, qb + 1u); } \
        else if ((S) == 2u) { W0 = __shfl_sync(0xffffffffu, (P).z, qb + 1u); W1 = __shfl_sync(0xffffffffu, (P).w, qb + 1u); W2 = __shfl_sync(0xffffffffu, (P).x, qb + 2u); } \
        else                { W0 = __shfl_sync(0xffffffffu, (P).y, qb + 2u); W1 = __shfl_sync(0xffffffffu, (P).z, qb + 2u); W2 = __shfl_sync(0xffffffffu, (P).w, qb + 2u); } } while (0)
    /* 第 (s,p) 组: 取码字 → 查表 + 激活读(发出去, 下一组再用) */
    #define TM_ISSUE(S, P, CA, CB, X) do { \
        const uint32_t w0_ = w[S][0], w1_ = w[S][1], w2_ = w[S][2]; \
        uint32_t va_ = (P) == 0u ? (__funnelshift_r(q3 ? w1_ : w0_, q3 ? w2_ : w1_, sh0) & 0xFFFu) : (__funnelshift_r(q2 ? w2_ : w1_, q2 ? 0u : w2_, sh1) & 0xFFFu); \
        if (EXT) va_ |= ((xa >> (8u * (S) + 4u * (P) + q)) & 1u) << 12; \
        CA = tm_word<EXT>(cbs, va_); \
        if (!HALF) { const uint32_t v0_ = v[S][0], v1_ = v[S][1], v2_ = v[S][2]; \
            uint32_t vb_ = (P) == 0u ? (__funnelshift_r(q3 ? v1_ : v0_, q3 ? v2_ : v1_, sh0) & 0xFFFu) : (__funnelshift_r(q2 ? v2_ : v1_, q2 ? 0u : v2_, sh1) & 0xFFFu); \
            if (EXT) vb_ |= ((xb >> (8u * (S) + 4u * (P) + q)) & 1u) << 12; \
            CB = tm_word<EXT>(cbs, vb_); } \
        X = *(const uint4 *)(xrow + (256u * c + 64u * (S) + 32u * (P) + 8u * q) / 2u); } while (0)
    #define TM_MMA(CA, CB, X) do { \
        uint32_t a_[4]; float t_[4]; \
        a_[0] = (CA).x; a_[1] = HALF ? 0u : (CB).x; a_[2] = (CA).y; a_[3] = HALF ? 0u : (CB).y; \
        tm_mma0(t_, a_, (X).x, (X).y); acc[0] += t_[0]; acc[1] += t_[1]; acc[2] += t_[2]; acc[3] += t_[3]; \
        a_[0] = (CA).z; a_[1] = HALF ? 0u : (CB).z; a_[2] = (CA).w; a_[3] = HALF ? 0u : (CB).w; \
        tm_mma0(t_, a_, (X).z, (X).w); acc[0] += t_[0]; acc[1] += t_[1]; acc[2] += t_[2]; acc[3] += t_[3]; } while (0)
    TM_PF(0u); TM_PF(1u); TM_PF(2u); TM_PF(3u);
    uint4 pa = make_uint4(0u, 0u, 0u, 0u), pb = pa, pa1 = pa, pb1 = pa, pa2 = pa, pb2 = pa;
    uint32_t xa = 0u, xb = 0u, xa1 = 0u, xb1 = 0u, xa2 = 0u, xb2 = 0u;
    TM_LD(0u, pa, pb, xa, xb); TM_LD(1u, pa1, pb1, xa1, xb1);
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    for (uint32_t c = 0; c < NCH; c++) {
        TM_LD(c + 2u, pa2, pb2, xa2, xb2);   /* 位流提前两块 */
        TM_PF(c + 4u);
        uint32_t w[4][3], v[4][3];
        TM_SHF(0u, pa, w[0][0], w[0][1], w[0][2]); TM_SHF(1u, pa, w[1][0], w[1][1], w[1][2]);
        TM_SHF(2u, pa, w[2][0], w[2][1], w[2][2]); TM_SHF(3u, pa, w[3][0], w[3][1], w[3][2]);
        if (!HALF) { TM_SHF(0u, pb, v[0][0], v[0][1], v[0][2]); TM_SHF(1u, pb, v[1][0], v[1][1], v[1][2]);
                     TM_SHF(2u, pb, v[2][0], v[2][1], v[2][2]); TM_SHF(3u, pb, v[3][0], v[3][1], v[3][2]); }
        uint4 ca0, cb0 = make_uint4(0u, 0u, 0u, 0u), x0, ca1, cb1 = cb0, x1;
        TM_ISSUE(0u, 0u, ca0, cb0, x0);
        TM_ISSUE(0u, 1u, ca1, cb1, x1); TM_MMA(ca0, cb0, x0);
        TM_ISSUE(1u, 0u, ca0, cb0, x0); TM_MMA(ca1, cb1, x1);
        TM_ISSUE(1u, 1u, ca1, cb1, x1); TM_MMA(ca0, cb0, x0);
        TM_ISSUE(2u, 0u, ca0, cb0, x0); TM_MMA(ca1, cb1, x1);
        TM_ISSUE(2u, 1u, ca1, cb1, x1); TM_MMA(ca0, cb0, x0);
        TM_ISSUE(3u, 0u, ca0, cb0, x0); TM_MMA(ca1, cb1, x1);
        TM_ISSUE(3u, 1u, ca1, cb1, x1); TM_MMA(ca0, cb0, x0);
        TM_MMA(ca1, cb1, x1);
        pa = pa1; pb = pb1; xa = xa1; xb = xb1; pa1 = pa2; pb1 = pb2; xa1 = xa2; xb1 = xb2;   /* 轮转: 下一块已在寄存器里 */
    }
    #undef TM_PF
    #undef TM_LD
    #undef TM_SHF
    #undef TM_ISSUE
    #undef TM_MMA
}
#define TM_SEG(U, UEND, ROWS, P, R0, N) \
    const uint32_t P = (U) / (ROWS), R0 = (U) % (ROWS); \
    uint32_t N = (UEND) - (U); if (N > (ROWS) - R0) N = (ROWS) - R0; if (N > 16u) N = 16u
__device__ __forceinline__ static float tm_gain(const mat_t &m, uint32_t r) { __half gh; memcpy(&gh, m.gr + (size_t)r * 2u, 2); return __half2float(gh); }

typedef struct { const uint8_t *g[NUMAX], *u[NUMAX], *d[NUMAX]; } exts_t;   /* 13 位层的位平面(每矩阵 rows × NI/8 B) */

/* gateup: 工作项 (组, 行), 组 ≤ 8 个 token; 出 h[pr][MID] = swiglu(bf16r(g·gain), bf16r(u·gain)) */
template <int EXT>
__global__ static void __launch_bounds__(TM_NW * 32u, 1) gu_mma(uint16_t *h, const mats_t *__restrict__ mt, const exts_t *__restrict__ ex, const int32_t *sel,
                                                        const int32_t *order, const uint32_t *x, uint32_t np, const uint8_t *cb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    tm_stage_cb<EXT>(vqsh, cb, EXT ? 8192u : 4096u);
    groups<(int)TM_MAXTOK>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    ts_begin();
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, g = lane >> 2;
    const uint32_t gw = blockIdx.x * TM_NW + (threadIdx.x >> 5), nw = gridDim.x * TM_NW;
    const uint32_t total = ng * MID, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        TM_SEG(u, uend, MID, grp, r0, n);
        u += n;
        const uint32_t qq = gq[grp], nt = gm[grp];
        const int32_t e = sel[order[qq]];
        const uint32_t *xrow = x + (uint64_t)((uint32_t)order[qq + (g < nt ? g : 0u)] / K) * (IN / 2u);   /* 本 lane 的 token 行(越界 lane 读 0 号, 不写) */
        const mat_t mg = MAT(mt, e, 0u), mu = MAT(mt, e, 1u);
        float ag[4], au[4];
        if (n <= 8u) { tm_tile<EXT, NIDX, 1>(mg, EXT ? ex->g[e] : NULL, r0, n, xrow, vqsh, ag); tm_tile<EXT, NIDX, 1>(mu, EXT ? ex->u[e] : NULL, r0, n, xrow, vqsh, au); }
        else         { tm_tile<EXT, NIDX, 0>(mg, EXT ? ex->g[e] : NULL, r0, n, xrow, vqsh, ag); tm_tile<EXT, NIDX, 0>(mu, EXT ? ex->u[e] : NULL, r0, n, xrow, vqsh, au); }
        const uint32_t la = r0 + g, lb = la + 8u;
        const float gla = tm_gain(mg, g < n ? la : r0), glb = tm_gain(mg, g + 8u < n ? lb : r0);   /* gate/up 同一组行增益(bench 两矩阵各带一份, 取 gate 的 = 引擎各取各的, 值同) */
        const float ula = tm_gain(mu, g < n ? la : r0), ulb = tm_gain(mu, g + 8u < n ? lb : r0);
        #pragma unroll
        for (uint32_t t = 0; t < 2u; t++) {
            const uint32_t tok = 2u * q + t;
            if (tok >= nt) continue;
            const uint64_t o = (uint64_t)(uint32_t)order[qq + tok] * MID;
            if (g < n) h[o + la] = swiglu(bf16r(ag[t] * gla), bf16r(au[t] * ula));
            if (g + 8u < n) h[o + lb] = swiglu(bf16r(ag[2u + t] * glb), bf16r(au[2u + t] * ulb));
        }
    }
    ts_end();
}
/* down: 工作项 (组, OUT 行); 出 partial[pr][OUT] = bf16r(y·gain) */
template <int EXT>
__global__ static void __launch_bounds__(TM_NW * 32u, 1) dn_mma(float *partial, const dmats_t *__restrict__ mt, const exts_t *__restrict__ ex, const int32_t *sel,
                                                        const int32_t *order, const uint32_t *hh, uint32_t np, const uint8_t *cb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    tm_stage_cb<EXT>(vqsh, cb, EXT ? 8192u : 4096u);
    groups<(int)TM_MAXTOK>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    ts_begin();
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, g = lane >> 2;
    const uint32_t gw = blockIdx.x * TM_NW + (threadIdx.x >> 5), nw = gridDim.x * TM_NW;
    const uint32_t total = ng * OUT, per = (total + nw - 1u) / nw;
    uint32_t u = gw * per;
    const uint32_t uend = (u + per < total) ? u + per : total;
    while (u < uend) {
        TM_SEG(u, uend, OUT, grp, r0, n);
        u += n;
        const uint32_t qq = gq[grp], nt = gm[grp];
        const int32_t e = sel[order[qq]];
        const uint32_t *hrow = hh + (uint64_t)(uint32_t)order[qq + (g < nt ? g : 0u)] * (MID / 2u);
        const mat_t md = DMAT(mt, e);
        float ad[4];
        if (n <= 8u) tm_tile<EXT, NIDX_DN, 1>(md, EXT ? ex->d[e] : NULL, r0, n, hrow, vqsh, ad);
        else         tm_tile<EXT, NIDX_DN, 0>(md, EXT ? ex->d[e] : NULL, r0, n, hrow, vqsh, ad);
        const uint32_t la = r0 + g, lb = la + 8u;
        const float gla = tm_gain(md, g < n ? la : r0), glb = tm_gain(md, g + 8u < n ? lb : r0);
        #pragma unroll
        for (uint32_t t = 0; t < 2u; t++) {
            const uint32_t tok = 2u * q + t;
            if (tok >= nt) continue;
            const uint64_t o = (uint64_t)(uint32_t)order[qq + tok] * OUT;
            if (g < n) partial[o + la] = bf16r(ad[t] * gla);
            if (g + 8u < n) partial[o + lb] = bf16r(ad[2u + t] * glb);
        }
    }
    ts_end();
}

/* ---- 参照: 同一 mma、最直白的实现(一个 block = 一个 warp 管一个 (组, 16 行瓦片), 码字逐个从全局按位取, 不分块不 shfl 不预取) ---- */
template <int EXT, uint32_t NI>
__device__ __forceinline__ static uint32_t tm_ref_code(const uint8_t *row, const uint8_t *exrow, uint32_t c) {   /* 第 c 个码字(12 位主流 + 位平面) */
    const uint32_t bit = 12u * c, byte = bit >> 3, sh = bit & 7u;
    uint32_t w; memcpy(&w, row + byte, 4);   /* 行尾多读的字节在下一行里(bench 载荷连续), 只取 12 位 */
    uint32_t v = (w >> sh) & 0xFFFu;
    if (EXT) v |= ((exrow[c >> 3] >> (c & 7u)) & 1u) << 12;
    return v;
}
template <int EXT, uint32_t NI>
__device__ __forceinline__ static void tm_ref_tile(const mat_t &m, const uint8_t *exm, uint32_t r0, uint32_t n, const uint32_t *xrow,
                                                   const uint8_t *cbs, float *acc) {
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, g = lane >> 2, h = q >> 1, hf = q & 1u;
    const uint32_t ra = r0 + (g < n ? g : n - 1u), rb = r0 + (g + 8u < n ? g + 8u : n - 1u);
    const uint8_t *rowa = m.ix + (size_t)ra * m.rowb, *rowb = m.ix + (size_t)rb * m.rowb;
    const uint8_t *exa = EXT ? exm + (size_t)ra * (NI / 8u) : NULL, *exb = EXT ? exm + (size_t)rb * (NI / 8u) : NULL;
    (void)h; (void)hf;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    for (uint32_t p = 0; p < NI / 4u; p++) {   /* 每 4 个码字: lane q 取码字 4p+q, 元素 0..3 当第 2p 组、4..7 当第 2p+1 组(与 tm_tile 同分组同序) */
        const uint32_t c = 4u * p + q;
        const uint4 ca = tm_word<EXT>(cbs, tm_ref_code<EXT, NI>(rowa, exa, c)), cb4 = tm_word<EXT>(cbs, tm_ref_code<EXT, NI>(rowb, exb, c));
        const uint4 x = *(const uint4 *)(xrow + (8u * c) / 2u);
        uint32_t a[4]; float t[4];
        a[0] = ca.x; a[1] = cb4.x; a[2] = ca.y; a[3] = cb4.y;
        tm_mma0(t, a, x.x, x.y);
        acc[0] += t[0]; acc[1] += t[1]; acc[2] += t[2]; acc[3] += t[3];
        a[0] = ca.z; a[1] = cb4.z; a[2] = ca.w; a[3] = cb4.w;
        tm_mma0(t, a, x.z, x.w);
        acc[0] += t[0]; acc[1] += t[1]; acc[2] += t[2]; acc[3] += t[3];
    }
}
template <int EXT, int MODE>   /* MODE 0 = gateup → h; 2 = down → partial。grid = (组数 × 瓦片数) 个 block, 每 block 32 线程 */
__global__ static void ref_mma(uint16_t *h, float *partial, const mats_t *__restrict__ mt, const dmats_t *__restrict__ dt, const exts_t *__restrict__ ex,
                               const int32_t *sel, const int32_t *order, const uint32_t *x, uint32_t np, const uint8_t *cb) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    tm_stage_cb<EXT>(vqsh, cb, EXT ? 8192u : 4096u);
    groups<(int)TM_MAXTOK>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    const uint32_t ROWS = MODE == 0 ? MID : OUT, ntile = ROWS / 16u;
    const uint32_t grp = blockIdx.x / ntile, r0 = (blockIdx.x % ntile) * 16u;
    if (grp >= ng) return;
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, g = lane >> 2;
    const uint32_t qq = gq[grp], nt = gm[grp];
    const int32_t e = sel[order[qq]];
    const uint32_t *xrow = x + (uint64_t)(MODE == 0 ? (uint32_t)order[qq + (g < nt ? g : 0u)] / K * (IN / 2u)
                                                    : (uint32_t)order[qq + (g < nt ? g : 0u)] * (MID / 2u));
    const uint32_t la = r0 + g, lb = la + 8u;
    if (MODE == 0) {
        const mat_t mg = MAT(mt, e, 0u), mu = MAT(mt, e, 1u);
        float ag[4], au[4];
        tm_ref_tile<EXT, NIDX>(mg, EXT ? ex->g[e] : NULL, r0, 16u, xrow, vqsh, ag);
        tm_ref_tile<EXT, NIDX>(mu, EXT ? ex->u[e] : NULL, r0, 16u, xrow, vqsh, au);
        for (uint32_t t = 0; t < 2u; t++) {
            const uint32_t tok = 2u * q + t;
            if (tok >= nt) continue;
            const uint64_t o = (uint64_t)(uint32_t)order[qq + tok] * MID;
            h[o + la] = swiglu(bf16r(ag[t] * tm_gain(mg, la)), bf16r(au[t] * tm_gain(mu, la)));
            h[o + lb] = swiglu(bf16r(ag[2u + t] * tm_gain(mg, lb)), bf16r(au[2u + t] * tm_gain(mu, lb)));
        }
    } else {
        const mat_t md = DMAT(dt, e);
        float ad[4];
        tm_ref_tile<EXT, NIDX_DN>(md, EXT ? ex->d[e] : NULL, r0, 16u, xrow, vqsh, ad);
        for (uint32_t t = 0; t < 2u; t++) {
            const uint32_t tok = 2u * q + t;
            if (tok >= nt) continue;
            const uint64_t o = (uint64_t)(uint32_t)order[qq + tok] * OUT;
            partial[o + la] = bf16r(ad[t] * tm_gain(md, la));
            partial[o + lb] = bf16r(ad[2u + t] * tm_gain(md, lb));
        }
    }
}
