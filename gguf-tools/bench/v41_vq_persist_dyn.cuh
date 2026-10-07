/* v41_vq_persist_dyn.cuh — 验证批常驻核(V0 抄本)的动态领工作形态(2026-10-07 晚), 被 v41_vq_persist_n_bench.cu 在 V0 核之后包含。
 *
 * 【为什么】打点(v41_vq_persist_ts.cuh)量到: n=4 时 V0 gu 核各 warp 行数相等, 完成时刻却差 254 µs(最快 ~350, 最慢 ~600), block 终点差 152 µs;
 * 两条机理: ① m=2 组(同一专家被两个 token 选中)的行每轮多一套 bf16 拆包 + FMA, 实测贵 18%(gu)/11%(dn); ② 等行静态划分下最后几个 warp
 * 独自收尾, SM 的访存管线没人填 —— 这正是引擎 nsys 里 gu 尾巴 142 µs × 40 层 = 5.7 ms/轮 的来源(dn 的 block 只能等 gu 整网格退完)。
 * 【怎么改】工作项 (组, 行) 空间不变, 不再按 warp 等分, 而是全核一个原子计数器按 U 行一单元发放: 谁先做完谁先领, 收尾离散度 ≤ 一个单元。
 * 单元内仍按 ≤32 行、不跨组切段(SEG), 每行的乘加序/规约树一个字不动 ⇒ 输出与 V0 逐字节同(门 = cmp)。
 * 第二版: 领单元提前一个(做本单元时下一单元号已知), 本单元末流的"next"指向下一单元首行 ⇒ 下一单元首块在本单元末块算完前就发出去,
 * 单元再小也不付"每单元一次 DRAM 延迟"(第一版 U=2 比 U=4 慢就是这笔钱)。单元跨组时第二段首块现取(U 小时极少见)。
 * 计数器由主机每发清零(引擎里交给前面的 order 核顺手清, 不多一发)。 */
template <int M, uint32_t U>
__global__ static void __launch_bounds__(1024, 1) gu_persist_dyn(uint16_t *h, const mats_t *__restrict__ mt, const int32_t *sel, const int32_t *order,
                                                                const uint32_t *x, uint32_t np, const uint8_t *cb, uint32_t *ctr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < CBB / 16u; i += blockDim.x) ((uint4 *)vqsh)[i] = ((const uint4 *)cb)[i];
    groups<M>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    ts_begin();
    const uint32_t total = ng * MID;
    uint32_t u = 0;
    if (lane == 0) u = atomicAdd(ctr, U);
    u = __shfl_sync(0xffffffffu, u, 0);
    blk_t carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u;
    if (u < total) carry = row_first_blk(MAT(mt, sel[order[gq[u / MID]]], 0u), u % MID);   /* 首单元首块 */
    while (u < total) {
        uint32_t unext = 0;
        if (lane == 0) unext = atomicAdd(ctr, U);   /* 先领下一单元 */
        unext = __shfl_sync(0xffffffffu, unext, 0);
        const uint32_t *nextp = unext < total ? row_ptr(MAT(mt, sel[order[gq[unext / MID]]], 0u), unext % MID) : NULL;   /* 下一单元首行(gate) */
        const uint32_t uend = (u + U < total) ? u + U : total;
        bool first = true;
        while (u < uend) {
            SEG(u, uend, MID, g, r0, n);
            u += n;
            const uint32_t q = gq[g], nt = gm[g];
            const int32_t e = sel[order[q]];
            const mat_t mg = MAT(mt, e, 0u), mu = MAT(mt, e, 1u);
            const uint32_t *xs[M]; uint32_t pr[M];
            #pragma unroll
            for (int j = 0; j < M; j++) { pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)]; xs[j] = x + (uint64_t)(pr[j] / K) * (IN / 2u); }
            if (!first) carry = row_first_blk(mg, r0);   /* 单元跨组: 第二段首块现取 */
            first = false;
            float gs[M], us[M];
            stream_m<M, NIDX>(mg, r0, n, xs, nt, vqsh, &carry, row_ptr(mu, r0), gs);
            stream_m<M, NIDX>(mu, r0, n, xs, nt, vqsh, &carry, u < uend ? NULL : nextp, us);   /* 单元末段: 顺手装下一单元首块 */
            if (lane < n) {
                #pragma unroll
                for (int j = 0; j < M; j++) if ((uint32_t)j < nt) h[(uint64_t)pr[j] * MID + r0 + lane] = swiglu(gs[j], us[j]);
            }
        }
        u = unext;
    }
    ts_end();
}
template <int M, uint32_t U>
__global__ static void __launch_bounds__(1024, 1) dn_persist_dyn(float *partial, const dmats_t *__restrict__ mt, const int32_t *sel, const int32_t *order,
                                                                const uint32_t *hh, uint32_t np, const uint8_t *cb, uint32_t *ctr) {
    extern __shared__ __align__(16) uint8_t vqsh[];
    __shared__ uint32_t gq[MAXP], gm[MAXP], ng;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < CBB / 16u; i += blockDim.x) ((uint4 *)vqsh)[i] = ((const uint4 *)cb)[i];
    groups<M>(sel, order, np, gq, gm, &ng);
    __syncthreads();
    ts_begin();
    const uint32_t total = ng * OUT;
    uint32_t u = 0;
    if (lane == 0) u = atomicAdd(ctr, U);
    u = __shfl_sync(0xffffffffu, u, 0);
    blk_t carry; carry.w0 = 0u; carry.w1 = 0u; carry.w2 = 0u;
    if (u < total) carry = row_first_blk(DMAT(mt, sel[order[gq[u / OUT]]]), u % OUT);
    while (u < total) {
        uint32_t unext = 0;
        if (lane == 0) unext = atomicAdd(ctr, U);
        unext = __shfl_sync(0xffffffffu, unext, 0);
        const uint32_t *nextp = unext < total ? row_ptr(DMAT(mt, sel[order[gq[unext / OUT]]]), unext % OUT) : NULL;
        const uint32_t uend = (u + U < total) ? u + U : total;
        bool first = true;
        while (u < uend) {
            SEG(u, uend, OUT, g, r0, n);
            u += n;
            const uint32_t q = gq[g], nt = gm[g];
            const int32_t e = sel[order[q]];
            uint32_t pr[M]; const uint32_t *hs[M];
            #pragma unroll
            for (int j = 0; j < M; j++) { pr[j] = (uint32_t)order[q + ((uint32_t)j < nt ? (uint32_t)j : 0u)]; hs[j] = hh + (uint64_t)pr[j] * (MID / 2u); }
            const mat_t md = DMAT(mt, e);
            if (!first) carry = row_first_blk(md, r0);
            first = false;
            float ys[M];
            stream_m<M, NIDX_DN>(md, r0, n, hs, nt, vqsh, &carry, u < uend ? NULL : nextp, ys);
            if (lane < n) {
                #pragma unroll
                for (int j = 0; j < M; j++) if ((uint32_t)j < nt) partial[(uint64_t)pr[j] * OUT + r0 + lane] = bf16r(ys[j]);
            }
        }
        u = unext;
    }
    ts_end();
}
