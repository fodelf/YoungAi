/* cuda_vq_fused2_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * fused2 特化高速 VQ 解码路。
 */
/* 双位流合并点积(gateup 专用): 同一行 m 的 w1/w3 两条独立 gather 链在一个循环里
 * 交替解码+乘加 —— 16 路在飞的 L1 访存互相填延迟(单链版每次只有 8 路, kernel 是
 * 延迟受限: ncu L1TEX scoreboard ~74%)。a0/a1 各自的求和顺序与两次单链调用完全
 * 相同 ⇒ 输出逐 bit 不变。 */
__device__ static void vq_row_dot2_dev(
        const uint8_t *blob, int e, uint32_t row, const float *vec, uint32_t lane,
        uint32_t *bs1, uint32_t *bs3, const __half *cbs1, const __half *cbs3,
        float *out_g, float *out_u) {
    *out_g = 0.0f; *out_u = 0.0f;
    const uint64_t off1 = vq_slot_dev(blob, e, 0);
    const uint64_t off3 = vq_slot_dev(blob, e, 1);
    if (!off1 || !off3) return;
    const uint8_t *pay1 = blob + off1, *pay3 = blob + off3;
    uint32_t mg1, mg3; memcpy(&mg1, pay1, 4); memcpy(&mg3, pay3, 4);
    if (mg1 != DS4VQ_MAT_MAGIC || mg3 != DS4VQ_MAT_MAGIC) return;
    uint16_t d16, n16; memcpy(&d16, pay1 + 4, 2); memcpy(&n16, pay1 + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay1 + 8, 4); memcpy(&cols, pay1 + 12, 4);
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    const uint8_t *cb1g = pay1 + 16, *cb3g = pay3 + 16;
    const uint8_t *cb1 = cbs1 ? (const uint8_t *)cbs1 : cb1g;
    const uint8_t *cb3 = cbs3 ? (const uint8_t *)cbs3 : cb3g;
    const uint8_t *gr1 = cb1g + (size_t)n16 * d16 * 2, *gr3 = cb3g + (size_t)n16 * d16 * 2;
    const uint8_t *ix1 = gr1 + (size_t)rows * 2, *ix3 = gr3 + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32u) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / d16;
    __half gh1, gh3; memcpy(&gh1, gr1 + (size_t)row * 2, 2); memcpy(&gh3, gr3 + (size_t)row * 2, 2);
    const float g1 = __half2float(gh1), g3 = __half2float(gh3);
    const size_t i0 = (size_t)row * nidx_row;
    size_t bsb1 = 0, bsb3 = 0;
    {
        const size_t byte0 = (i0 * nbit) >> 3;
        bsb1 = bsb3 = byte0 & ~(size_t)3;
        const size_t byte_end = (((i0 + nidx_row) * nbit + 7u) >> 3) + 4u;
        uint32_t nw = (uint32_t)((byte_end - bsb1 + 3u) >> 2);
        if (nw > DS4_VQ_BITWORDS) nw = DS4_VQ_BITWORDS;
        for (uint32_t w = lane; w < nw; w += 32u) {
            memcpy(&bs1[w], ix1 + bsb1 + (size_t)w * 4u, 4);
            memcpy(&bs3[w], ix3 + bsb3 + (size_t)w * 4u, 4);
        }
        __syncwarp();
    }
    float a0 = 0.0f, a1 = 0.0f;
    uint32_t j = lane;
    for (; j + (DS4_VQ_BATCH - 1u) * 32u < nidx_row; j += DS4_VQ_BATCH * 32u) {
        uint32_t v1[DS4_VQ_BATCH], v3[DS4_VQ_BATCH];
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {
            const uint32_t jj = j + k * 32u;
            if (nbit == 8u) {
                v1[k] = ix1[i0 + jj];
                v3[k] = ix3[i0 + jj];
            } else {
                const size_t bit = (i0 + jj) * nbit;
                uint32_t w1_, w3_;
                memcpy(&w1_, (const uint8_t *)bs1 + ((bit >> 3) - bsb1), 4);
                memcpy(&w3_, (const uint8_t *)bs3 + ((bit >> 3) - bsb3), 4);
                v1[k] = (w1_ >> (bit & 7)) & imsk;
                v3[k] = (w3_ >> (bit & 7)) & imsk;
            }
        }
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {
            const float *vs = vec + (size_t)(j + k * 32u) * d16;
            if (d16 == 4u) {
                uint64_t q1, q3;
                memcpy(&q1, cb1 + (size_t)v1[k] * 8u, 8);
                memcpy(&q3, cb3 + (size_t)v3[k] * 8u, 8);
                #pragma unroll
                for (uint32_t d = 0; d < 4u; d++) {
                    __half h1, h3;
                    const uint16_t u1 = (uint16_t)(q1 >> (d * 16));
                    const uint16_t u3 = (uint16_t)(q3 >> (d * 16));
                    memcpy(&h1, &u1, 2); memcpy(&h3, &u3, 2);
                    a0 += __half2float(h1) * vs[d];
                    a1 += __half2float(h3) * vs[d];
                }
            } else {
                const uint8_t *c1 = cb1 + (size_t)v1[k] * d16 * 2;
                const uint8_t *c3 = cb3 + (size_t)v3[k] * d16 * 2;
                for (uint32_t d = 0; d < d16; d++) {
                    __half h1, h3;
                    memcpy(&h1, c1 + (size_t)d * 2, 2);
                    memcpy(&h3, c3 + (size_t)d * 2, 2);
                    a0 += __half2float(h1) * vs[d];
                    a1 += __half2float(h3) * vs[d];
                }
            }
        }
    }
    for (; j < nidx_row; j += 32u) {
        uint32_t v1, v3;
        if (nbit == 8u) {
            v1 = ix1[i0 + j]; v3 = ix3[i0 + j];
        } else {
            const size_t bit = (i0 + j) * nbit;
            uint32_t w1_, w3_;
            memcpy(&w1_, (const uint8_t *)bs1 + ((bit >> 3) - bsb1), 4);
            memcpy(&w3_, (const uint8_t *)bs3 + ((bit >> 3) - bsb3), 4);
            v1 = (w1_ >> (bit & 7)) & imsk;
            v3 = (w3_ >> (bit & 7)) & imsk;
        }
        const float *vs = vec + (size_t)j * d16;
        const uint8_t *c1 = cb1 + (size_t)v1 * d16 * 2;
        const uint8_t *c3 = cb3 + (size_t)v3 * d16 * 2;
        for (uint32_t d = 0; d < d16; d++) {
            __half h1, h3;
            memcpy(&h1, c1 + (size_t)d * 2, 2);
            memcpy(&h3, c3 + (size_t)d * 2, 2);
            a0 += __half2float(h1) * vs[d];
            a1 += __half2float(h3) * vs[d];
        }
    }
    *out_g = a0 * g1;
    *out_u = a1 * g3;
}

/* 把 (e,which) 矩阵的码本搬进 shared。同一 block 的所有 warp 处理同一个
 * (token,expert) 对 ⇒ 共用一本码本, 搬一次全 block 受益。容量不够返回 0 走全局。 */
__device__ __forceinline__ static int vq_load_cb_sh(
        const uint8_t *blob, int e, int which, __half *dst, uint32_t dst_cap_halfs) {
    const uint64_t off = vq_slot_dev(blob, e, which);
    if (off == 0u) return 0;
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return 0;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    const uint32_t n = (uint32_t)n16 * d16;
    if (n > dst_cap_halfs) return 0;
    const uint8_t *cb = pay + 16;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        __half h; memcpy(&h, cb + (size_t)i * 2, 2);
        dst[i] = h;
    }
    return 1;
}

__global__ static void vq_moe_gateup_fused_kernel(
        float *h, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *x, uint32_t n_expert, uint32_t IN, uint32_t MID, float clamp) {
    /* ★专家放最慢变化的维度★: CUDA 按 blockIdx.x 最快变化调度, 原来 pk 在 y、行块
     * 在 z, 于是同时在跑的 block 分属 6 个不同专家 —— 12 个内存流(6 专家×gate/up)
     * 在 DRAM 上交替跳, row buffer 全程冲突, 位流实测只跑出 4GB/s。
     * 改成行块在 y、专家在 z 后, 同时运行的 block 读同一专家的连续位流。 */
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0 || rw[pair] == 0.0f) return;
    /* 码本进 shared: 同一 block 的 8 个 warp 处理同一个 (token,expert) 对, 共用同
     * 一本码本(512×4×2=4KB)。随机查表落到片上后不再受 L1 tag/串行化限制 —— 这是
     * 当前 kernel 唯一的重瓶颈(每 4 元素一次随机 gather)。
     * (早前一次失败的尝试是把它和"多行/block"一起改, 两个变量混在一起, 这次只动这个。) */
    /* x 不进 shared: 实测零收益(它在 L1/L2 本就高命中), 而 16KB 会把 occupancy 压死。
     * shared 全部让给码本 —— 真正的瓶颈是位流的非合并读, 见 vq_row_dot_dev。 */
    /* shared 只给位流: 码本进 shared 实测零收益(它在片上本就高命中), 白占 8KB 反而
     * 压低 occupancy。位流的非合并读才是那 70%。 */
    extern __shared__ uint32_t shg[];
    const float *sx = x + (uint64_t)t * IN;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t *bs1 = shg + (size_t)warp * 2u * DS4_VQ_BITWORDS;
    uint32_t *bs3 = bs1 + DS4_VQ_BITWORDS;
    /* 码本进 shared(block 级一次): dot2 后 gather 密度翻倍, 片上码本免 L1 tag 竞争。
     * 布局: [warps*2*BITWORDS u32][cb1 512*4 half][cb3 512*4 half] */
    __half *cbs1 = (__half *)(shg + (size_t)DS4_VQ_WARPS_PER_BLOCK * 2u * DS4_VQ_BITWORDS);
    __half *cbs3 = cbs1 + 2048u;
    float *xsh = (float *)(cbs3 + 2048u);   /* x 向量 IN floats 进 shared */
    {
        const uint64_t o1_ = vq_slot_dev(blob, e, 0);
        const uint64_t o3_ = vq_slot_dev(blob, e, 1);
        const uint8_t *c1_ = blob + o1_ + 16, *c3_ = blob + o3_ + 16;
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint16_t u1_, u3_;
            memcpy(&u1_, c1_ + (size_t)i * 2u, 2);
            memcpy(&u3_, c3_ + (size_t)i * 2u, 2);
            memcpy(&cbs1[i], &u1_, 2);
            memcpy(&cbs3[i], &u3_, 2);
        }
        for (uint32_t i = threadIdx.x; i < IN; i += blockDim.x) xsh[i] = sx[i];
        __syncthreads();
    }
    const uint32_t m = blockIdx.y * DS4_VQ_WARPS_PER_BLOCK + warp;
    if (m >= MID) return;
    float g, u;
    vq_row_dot2_dev(blob, e, m, xsh, lane, bs1, bs3, cbs1, cbs3, &g, &u);
    g = vq_warp_reduce(g);
    u = vq_warp_reduce(u);
    if (lane == 0) {
        if (clamp > 0.0f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        h[pair * MID + m] = (g / (1.0f + __expf(-g))) * u;
    }
}

__global__ static void vq_moe_down_fused_kernel(
        float *partial, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *h, const uint8_t *down_base, uint64_t down_ebytes,
        uint32_t n_expert, uint32_t MID, uint32_t OUT) {
    /* per-pick 并行 + 独立 partial 平面直写(无 atomicAdd), 汇总由固定 pk 序的
     * vq2_down_reduce_kernel 完成 ⇒ 温 0 逐 bit 可复现且保留 6× 并行度(曾试
     * 单 warp 串行 6 pick 的确定化, down 从 ~0.4ms 涨到 ~1.2ms/层)。 */
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0) return;   /* 未写平面由 reduce 按 sel/rw 跳过 */
    const float w = rw[pair];
    if (w == 0.0f) return;
    extern __shared__ uint32_t shd[];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t *bs = shd + (size_t)warp * DS4_VQ_BITWORDS;
    /* w2 码本进 shared(block 级一次), 同 gateup。槽缺失(冷 w2)时跳过, dot 走冷路。 */
    __half *cbs2 = (__half *)(shd + (size_t)DS4_VQ_WARPS_PER_BLOCK * DS4_VQ_BITWORDS);
    const uint64_t o2_ = vq_slot_dev(blob, e, 2);
    if (o2_) {
        const uint8_t *c2_ = blob + o2_ + 16;
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint16_t u2_; memcpy(&u2_, c2_ + (size_t)i * 2u, 2);
            memcpy(&cbs2[i], &u2_, 2);
        }
    }
    __syncthreads();
    const uint32_t o = blockIdx.y * DS4_VQ_WARPS_PER_BLOCK + warp;
    if (o >= OUT) return;
    float acc = vq_row_dot_dev(blob, e, 2, o, h + pair * MID, lane, down_base, down_ebytes, MID,
                               o2_ ? cbs2 : NULL, bs);
    acc = vq_warp_reduce(acc);
    if (lane == 0) partial[pair * OUT + o] = w * acc;
}


