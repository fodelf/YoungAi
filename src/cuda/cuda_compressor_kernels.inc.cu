/* cuda_compressor_kernels.inc.cu — 压缩器投影"攒到 emit 一次算"的两个核(2026-09-05)。
 *
 * 为什么: 压缩器 kv/gate 投影是 f16 权重(21 层 ratio-4 各 16.8 MB + 20 层 ratio-128 各 8.4 MB
 * + 21 层 indexer 各 4.2 MB = 609 MB/token), 原来每个 token 都读一遍(2.6 ms/token, 贴带宽墙)。
 * 但压缩器 state 只是逐 token 投影的暂存区(compressor_store_kernel 按 pos%ratio 写行),
 * emit 时才池化 ⇒ 投影可以攒 n 个 token 在 emit 时一次算: 权重每读一遍算 8 个 token,
 * ratio-4 层权重读量降 4×, ratio-128 层降 16×(8 token 一趟, 16 趟)。
 * 数值铁律: 每行(每 token)的累加序必须与单 token 的 matmul_f16_pair_rowblock_kernel
 * (pair2=0 形态)逐字相同 —— 同 256 线程整行、同 uint4 读宽、同 8 项表达式、同 warp shuffle
 * 树 + 8 个 warp 定序求和。这里只是把 x 换成 x[t], 权重 uint4 复用给 8 个 token。改了会怎样:
 * 表达式或归约序一变, 压缩行就与 prefill/老解码路差 ULP, 再经 FP8 舍入放大(prefill_util
 * 里"混两种累加序翻 FP8 舍入"的先例), 对拍逐字节必分叉。
 */
#define DS4_F16_PAIR_TOKCHUNK 8u

__global__ static void matmul_f16_pair_rowblock_mtok_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,          /* [n_tok][in_dim] */
        uint64_t in_dim,         /* 必须 8 的倍数(调用方保证) */
        uint64_t out_dim,
        uint32_t n_tok) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t tid = threadIdx.x;
    __shared__ float wsum[8];
    for (uint32_t r = blockIdx.x; r < 2u * out_dim; r += gridDim.x) {
        const uint32_t which = (r >= out_dim) ? 1u : 0u;
        const uint64_t row = which ? (uint64_t)(r - out_dim) : (uint64_t)r;
        const uint4 *wr4 = (const uint4 *)((which ? w1 : w0) + row * in_dim);
        const uint64_t n8 = in_dim >> 3;
        for (uint32_t t0 = 0; t0 < n_tok; t0 += DS4_F16_PAIR_TOKCHUNK) {
            const uint32_t nt = n_tok - t0 < DS4_F16_PAIR_TOKCHUNK ? n_tok - t0 : DS4_F16_PAIR_TOKCHUNK;
            float sum[DS4_F16_PAIR_TOKCHUNK];
#pragma unroll
            for (uint32_t m = 0; m < DS4_F16_PAIR_TOKCHUNK; m++) sum[m] = 0.0f;
            for (uint64_t i = tid; i < n8; i += 256u) {
                const uint4 v = wr4[i];
                const __half2 *h = (const __half2 *)&v;
#pragma unroll
                for (uint32_t m = 0; m < DS4_F16_PAIR_TOKCHUNK; m++) {
                    if (m < nt) {
                        const float *xm = x + (uint64_t)(t0 + m) * in_dim;
                        const float4 xa = *(const float4 *)(xm + 8u * i);
                        const float4 xb = *(const float4 *)(xm + 8u * i + 4u);
                        sum[m] += __low2float(h[0]) * xa.x + __high2float(h[0]) * xa.y
                                + __low2float(h[1]) * xa.z + __high2float(h[1]) * xa.w
                                + __low2float(h[2]) * xb.x + __high2float(h[2]) * xb.y
                                + __low2float(h[3]) * xb.z + __high2float(h[3]) * xb.w;
                    }
                }
            }
#pragma unroll
            for (uint32_t m = 0; m < DS4_F16_PAIR_TOKCHUNK; m++) {
                float s = sum[m];
                for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
                if ((tid & 31u) == 0) wsum[tid >> 5u] = s;
                __syncthreads();
                if (tid == 0 && m < nt) {
                    float t = 0.0f;
                    for (uint32_t wgi = 0; wgi < 8u; wgi++) t += wsum[wgi];
                    (which ? out1 : out0)[(uint64_t)(t0 + m) * out_dim + row] = t;
                }
                __syncthreads();
            }
        }
    }
}

/* x 行进环: 16 KB 一行, 用核而不是 cudaMemcpyAsync —— 图重放时目标行随 pos 变, kernel 参数
 * 走 ExecUpdate 稳; memcpy 节点改指针不在同一条验证过的路上。 */
__global__ static void comp_ring_push_kernel(float4 *dst, const float4 *src, uint32_t n4) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}
