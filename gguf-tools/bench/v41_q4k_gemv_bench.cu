/* v41_q4k_gemv_bench.cu — V4.1 骨架 q4_K 解码 GEMV 的形态微基准(spark 本机跑; 2026-09-23 立, 09-24 扩到多行激活)。
 *
 * 【为什么要它】引擎一趟要装 113 GB 模型 + 4 分钟, 换核形态先在这里过, 过了才进引擎(sp.md 的规矩)。
 * 09-23: 纯解码(NT=1)从"每 warp 预取 4 块"换成 stage/pipe(CTA 先把连续几行搬进 shared), 196 → 219 GB/s 级。
 * 09-24: 投机验证批(NT = 1+k 行激活)也走 stage/pipe 之后, 真实请求 4 行 GEMV 仍 ~25.7 ms 对单行 ~18 ms。
 *   账: 每个 warp 算自己那一行时把同一段激活从 L1 重读一遍, 每个权重元素每行激活 4 B ⇒ NT=4 一步 ~115 GB 的 L1 流量,
 *   GB10 L1 合计 ~14.7 TB/s ⇒ ~7.8 ms, 正好是多出来的那截。NT=1 时 29 GB 藏在 DRAM 下面看不见。
 *
 * 【比什么】R = 一个 warp 管几行。R=1 = 引擎现核 stage/pipe<NT> 的同序重写(块序/段序/规约序与引擎一致; 真终验是引擎逐字节门)。
 *   R>1: CTA 的 warp 数 8 → 8/R, 每 warp 管 R 行(同一 K 段), 激活每块只读一次给 R 行用 ⇒ L1 流量 ÷R。
 *   CTA 负责的行/搬进 shared 的字节/每 SM 的 warp 总数都不变; 只在 CTA 行数(8/ksplit) ≥ R 时可用。
 * 判据(两条都要): ①各 R 输出与 R=1 逐位同 ②NT 行里第 t 行 == 只拿第 t 行激活跑 NT=1 的结果, 逐位同 ——
 *   这就是引擎"投机 == 纯解码逐字节"那条门在核层面的样子。不同 = 块序/段序被改了, 不许进引擎。
 * 形状 = v3 GGUF 的真实张量(09-23 从 GGUF 头读出), 见 main() 的 shapes 表。
 * 09-29: 改成比"发法"(引擎规则 / 一律 pipe×4 / 一律 pipe×8 / 一律 stage), R 固定 1(R>1 已判负)。逐形状 NT=1: q_b pipe 173 → stage 225 GB/s,
 *   共享专家 153 → 188, kv 124 → 160; 每步合计 引擎规则 24.35 → 全 stage 21.10 ms; NT=4 时 q_b 反而 pipe×4 快(199 vs 185)。
 *   ⇒ 引擎不再按 4~12 KB 档位定发法, 改成每 (形状, 行数) 第一次直发时自己量两种取快的(cuda_v41_q4k.inc.cu v41_q4k_pick)。
 * 用法: nvcc -O3 -arch=native -o v41_q4k_gemv_bench v41_q4k_gemv_bench.cu && ./v41_q4k_gemv_bench [iters=50] [NT=4] [只跑第几个形状=-1 全部] [只跑第几种发法=-1 全部] [PDL 发射=0] [L2 预取试验=0] [激活 0=f32 1=bf16 2=bf16块排布] [R=1]
 *   ★后两个参数给 ncu 用(09-29 晚)★: 全跑时每个形状还会为判据②多发 NT 次单行核, k_stage/k_pipe 的发射序号随 NT 变,
 *   靠 --launch-skip 数序号会数错核(实撞: 想量 wo_b NT=4 量到的是 wo_a 的单行核)。钉住形状与发法后 skip 5(暖身) count 1 就是那一发。 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define BLK 256u
#define BYTES 144u
#define WARPS 8u               /* 引擎 V41_GEMV_WARPS: ksplit 的上限, 也是 R=1 时一个 CTA 的 warp 数 */
#define PIPE_MIN 4096u
#define PIPE_MAX 12288u

/* ---- 引擎同款(逐字) ---- */
__device__ __forceinline__ static uint32_t scb(const uint4 &h, uint32_t k) {
    const uint32_t w = k < 4u ? h.y : (k < 8u ? h.z : h.w);
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}
__device__ __forceinline__ static void sm_reg(const uint4 &h, uint32_t j, float *s, float *m) {
    uint32_t a, b;
    if (j < 4u) { a = scb(h, j) & 63u; b = scb(h, j + 4u) & 63u; }
    else { a = (scb(h, j + 4u) & 0xFu) | ((scb(h, j - 4u) >> 6) << 4);
           b = (scb(h, j + 4u) >> 4)   | ((scb(h, j) >> 6) << 4); }
    *s = (float)a; *m = (float)b;
}
/* 一块解成本 lane 的 8 个权重(低 4 + 高 4), 与引擎 v41_q4k_blk_acc_reg 前半逐式同 */
__device__ __forceinline__ static void blk_w(const uint4 &h, uint32_t qw, float *wl, float *wh) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3;
    const float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFFu)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    float s_lo, m_lo, s_hi, m_hi;
    sm_reg(h, gidx * 2u, &s_lo, &m_lo);
    sm_reg(h, gidx * 2u + 1u, &s_hi, &m_hi);
    const float dl = d * s_lo, ml = dmin * m_lo, dh = d * s_hi, mh = dmin * m_hi;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t byte = (qw >> (8 * i)) & 0xFFu;
        wl[i] = dl * (float)(byte & 0xFu) - ml;
        wh[i] = dh * (float)(byte >> 4) - mh;
    }
}
/* 点积一项: 与引擎 acc[t] += wl[0]*xl.x + … + wh[3]*xh.w 同一棵表达式树(收缩成 FMA 的方式因此相同) */
__device__ __forceinline__ static float dot8(const float *wl, const float *wh, const float4 &xl, const float4 &xh) {
    return wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
         + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
}

/* 一个 warp: R 行(shared 里第 row0..row0+R-1 行) × 本 K 段 × NT 条激活。块按 b 升序、每行 acc[j][t] 逐块累加 ⇒ 与 R=1 同序。
 * 激活按 t 外层读一次(8 个 float), 给 R 行各乘一遍 —— 这就是省 L1 流量的地方。 */
/* ★XB=1: 激活按 bf16 读(2026-10-01)★: NT 行里每 lane 每块要读 NT×(4+4) 个激活, f32 是两条 float4 = 32 B/行, 32 lane 铺开 1 KB = 8 个 L1 波前;
 * 激活本来就在 bf16 格点上(调用方 rms_norm/bf16r 舍过, 与 fp4x32 骨架 NT>1 那条路同一前提), 存成 bf16 后两条 uint2 = 16 B/行 = 4 个波前 —— 这是
 * NT>1 多行税的大头(微基准 NT=6 +11.7 ms/步 ≈ 224M 块 × 8 波前 × 6 行 ÷ (48 SM × 2.42 GHz)), 对半砍。bf16→f32 补零精确, dot8 同一棵树 ⇒ 逐位同。 */
__device__ __forceinline__ static float4 bf16x4_to_f4(uint2 v) {
    float4 r;
    r.x = __uint_as_float(v.x << 16); r.y = __uint_as_float(v.x & 0xffff0000u);
    r.z = __uint_as_float(v.y << 16); r.w = __uint_as_float(v.y & 0xffff0000u);
    return r;
}
template <uint32_t NT, uint32_t R, uint32_t XB>
__device__ __forceinline__ static void rows_acc(const uint8_t *sw, uint32_t row0, uint32_t nrows_here, uint32_t nblk, uint32_t ksplit,
                                                uint32_t kpart, const void *x, uint32_t x_stride, float (*acc)[NT]) {
    const uint32_t lane = threadIdx.x & 31u, gidx = lane >> 3, q0 = (lane & 7u) * 4u;
    for (uint32_t b = kpart; b < nblk; b += ksplit) {
        float wl[R][4], wh[R][4];
        #pragma unroll
        for (uint32_t j = 0; j < R; j++) {
            if (row0 + j < nrows_here) {
                const uint8_t *blk = sw + ((uint64_t)(row0 + j) * nblk + b) * BYTES;
                blk_w(*(const uint4 *)blk, *(const uint32_t *)(blk + 16u + 4u * lane), wl[j], wh[j]);
            }
        }
        const uint32_t e_lo = b * BLK + gidx * 64u + q0, e_hi = e_lo + 32u;
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float4 xl, xh;
            if (XB == 2) {   /* 块友好排布: 第 b 块里 lane l 的 8 个 bf16 连续放在 b*256 + l*8 ⇒ 一条 16 B 读(指令数再砍半, 波前数与 XB=1 同) */
                const uint16_t *xt = (const uint16_t *)x + (uint64_t)t * x_stride;
                const uint4 v = *(const uint4 *)(xt + b * BLK + lane * 8u);
                uint2 lo, hi; lo.x = v.x; lo.y = v.y; hi.x = v.z; hi.y = v.w;
                xl = bf16x4_to_f4(lo); xh = bf16x4_to_f4(hi);
            } else if (XB) { const uint16_t *xt = (const uint16_t *)x + (uint64_t)t * x_stride;
                      xl = bf16x4_to_f4(*(const uint2 *)(xt + e_lo)); xh = bf16x4_to_f4(*(const uint2 *)(xt + e_hi)); }
            else { const float *xt = (const float *)x + (uint64_t)t * x_stride;
                   xl = *(const float4 *)(xt + e_lo); xh = *(const float4 *)(xt + e_hi); }
            #pragma unroll
            for (uint32_t j = 0; j < R; j++) if (row0 + j < nrows_here) acc[j][t] += dot8(wl[j], wh[j], xl, xh);
        }
    }
}
/* 规约与写出: warp xor → red[(行·ksplit + 段)·NT + t] → 段 0 那个 warp 按段号升序相加(与引擎同序) */
template <uint32_t NT, uint32_t R>
__device__ __forceinline__ static void rows_finish(float (*acc)[NT], float *out, uint32_t out_stride, uint32_t r0, uint32_t row0,
                                                   uint32_t nrows_here, uint32_t ksplit, uint32_t kpart, float *red) {
    const uint32_t lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t j = 0; j < R; j++)
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float a = acc[j][t];
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0 && row0 + j < nrows_here) red[((row0 + j) * ksplit + kpart) * NT + t] = a;
        }
    __syncthreads();
    if (kpart == 0 && lane == 0)
        #pragma unroll
        for (uint32_t j = 0; j < R; j++) {
            if (row0 + j >= nrows_here) continue;
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float sum = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) sum += red[((row0 + j) * ksplit + k) * NT + t];
                out[(uint64_t)t * out_stride + r0 + row0 + j] = sum;
            }
        }
}

/* ★PDL(09-29 晚)★: 引擎里这对核是编程式发射的(权重先搬进 shared, 再 griddepcontrol.wait 等上游, 见 cuda_internal.cuh);
 * 微基准以前是普通发射, 于是"下一发能不能趁上一发收尾时提前上车"这件事根本量不到 —— 09-29 中午发法 A/B 在引擎里翻车就是这个盲区。
 * 这里照引擎放 wait, 之后紧跟 launch_dependents(引擎走图用的是 LaunchCompletion 端口 = 上游所有 block 都已开跑, 这里用触发来等价它)。
 * 普通发射时两句都是空转, 老读数不变。 */
__device__ __forceinline__ static void pdl_wait_trigger() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
}
/* stage: 一个 CTA 一组行(rpb = WARPS/ksplit 行, 连续一段), 整段搬进 shared 再算。blockDim = 32·WARPS/R。
 * MINB = __launch_bounds__ 的每 SM 最少 CTA 数 = 编译器的寄存器上限(65536 / (256·MINB)): 4 ⇒ 64 个, 5 ⇒ 51(按 8 取整 48)。
 * ★为什么要比 4 与 5★: 引擎 NT=1 只用 43 个寄存器, 每 SM 实际放得下 5 个 CTA —— 上一发还剩尾波时下一发已能占第 5 个空位预搬权重;
 * NT≥3 编译器把 64 个用满, 空位归零, PDL 的预搬无处可放。这个假设只有带 PDL 的链式发射量得出来。 */
template <uint32_t NT, uint32_t R, uint32_t MINB, uint32_t XB>
__global__ static void __launch_bounds__(256 / R, MINB * R) k_stage(float *out, const uint8_t *w, const void *x, uint32_t in_dim,
                                                                    uint32_t out_dim, uint32_t ksplit, uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 st[];
    __shared__ float red[WARPS * NT];
    const uint32_t warp = threadIdx.x >> 5, rpb = WARPS / ksplit, nblk = in_dim / BLK;
    const uint32_t kpart = warp % ksplit, row0 = (warp / ksplit) * R;
    const uint32_t r0 = blockIdx.x * rpb, nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb, n16 = nrows * nblk * (BYTES / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BYTES);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) st[i] = __ldcs(src + i);
    pdl_wait_trigger();
    __syncthreads();
    float acc[R][NT];
    #pragma unroll
    for (uint32_t j = 0; j < R; j++)
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) acc[j][t] = 0.f;
    rows_acc<NT, R, XB>((const uint8_t *)st, row0, nrows, nblk, ksplit, kpart, x, x_stride, acc);
    rows_finish<NT, R>(acc, out, out_stride, r0, row0, nrows, ksplit, kpart, red);
}
/* pipe: CTA 常驻, 按 rg += gridDim.x 循环行组, cp.async 双缓冲(算第 k 组时搬第 k+1 组) */
template <uint32_t NT, uint32_t R, uint32_t OCC, uint32_t XB>
__global__ static void __launch_bounds__(256 / R, OCC * R) k_pipe(float *out, const uint8_t *w, const void *x, uint32_t in_dim,
                                                                uint32_t out_dim, uint32_t ksplit, uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 st[];
    __shared__ float red[WARPS * NT];
    const uint32_t warp = threadIdx.x >> 5, rpb = WARPS / ksplit, nblk = in_dim / BLK;
    const uint32_t kpart = warp % ksplit, row0 = (warp / ksplit) * R;
    const uint32_t ngrp = (out_dim + rpb - 1u) / rpb, grp16 = rpb * nblk * (BYTES / 16u);
    auto issue = [&](uint32_t rg, uint32_t buf) {
        const uint32_t r0 = rg * rpb, nr = out_dim - r0 < rpb ? out_dim - r0 : rpb, n16 = nr * nblk * (BYTES / 16u);
        const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BYTES);
        uint4 *dst = st + (uint64_t)buf * grp16;
        for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) __pipeline_memcpy_async(dst + i, src + i, 16);
    };
    uint32_t rg = blockIdx.x, buf = 0;
    if (rg < ngrp) issue(rg, 0);
    __pipeline_commit();
    pdl_wait_trigger();   /* 引擎同位置: 第一组权重已在飞, 这里才等上游的激活 */
    for (; rg < ngrp; rg += gridDim.x, buf ^= 1u) {
        if (rg + gridDim.x < ngrp) issue(rg + gridDim.x, buf ^ 1u);
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        const uint32_t r0 = rg * rpb, nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb;
        float acc[R][NT];
        #pragma unroll
        for (uint32_t j = 0; j < R; j++)
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) acc[j][t] = 0.f;
        rows_acc<NT, R, XB>((const uint8_t *)(st + (uint64_t)buf * grp16), row0, nrows, nblk, ksplit, kpart, x, x_stride, acc);
        rows_finish<NT, R>(acc, out, out_stride, r0, row0, nrows, ksplit, kpart, red);
        __syncthreads();
    }
}

typedef struct { const char *name; uint32_t out_dim, in_dim, per_step, groups; } shape_t;   /* groups 只用于按引擎口径算 ksplit */

/* ★L2 预取存活试验(09-29 晚)★: 引擎 n=4 验证步里注意力核每层独占 ~104 µs, 零 DRAM 字节, 期间 DRAM 空转; 紧接着的 wo_a(19 MB)
 * 要读 ~100 µs。想法 = 注意力跑的时候把 wo_a 预取进 L2(24 MB), 之后 GEMV 命中 L2。09-23 在引擎里试过一版判负(慢 1.7 ms/步):
 * 每步多 271 个预取节点, 且消费方 __ldcs(evict-first)流式读把 L2 冲掉, 预取行等不到被用。这里先在孤立尺上回答两个问题:
 * ①预取带 evict_last 策略、中间夹一个只算不读的核(假注意力)之后, GEMV 还命中 L2 吗(GEMV 时间掉多少) ②预取核自己要多久。
 * 都过了才值得改引擎(预取指令放进注意力核的 block 里, 零额外节点)。 */
__global__ static void k_prefetch(const uint8_t *p, uint64_t bytes, int how) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, nth = (uint64_t)gridDim.x * blockDim.x;
    if (how == 1) {   /* TMA 批量预取, 每线程 4 KB 一条, 带 evict_last 缓存策略 */
        uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
        for (uint64_t off = tid * 4096u; off < bytes; off += nth * 4096u) {
            const uint32_t sz = (uint32_t)((bytes - off < 4096u) ? ((bytes - off) & ~15ull) : 4096u);
            if (sz) asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p + off), "r"(sz), "l"(pol) : "memory");
        }
    } else {          /* 逐 128 B 线 prefetch.global.L2::evict_last */
        for (uint64_t off = tid * 128u; off < bytes; off += nth * 128u) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + off));
    }
}
/* 假注意力: 先流式读一遍 scratch(模拟 KV 读 ~2 MB), 再纯算到 cycles 个时钟(模拟 ~100 µs 的零字节段) */
__global__ static void k_dummy(const float *scratch, uint64_t n, float *sink, long long cycles) {
    float a = 0.f;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) a += scratch[i];
    const long long t0 = clock64();
    float b = a;
    while (clock64() - t0 < cycles) { for (int k = 0; k < 64; k++) b = b * 1.000001f + 0.5f; }
    if (b == 12345.f) *sink = b;
}
static int g_l2pf = 0;
static int g_xb = 0, g_r = 1;   /* 第 8 个参数: R = 一个 warp 管几行(激活每块读一次给 R 行用; CTA 行数 ≥ R 才用) */     /* 第 7 个参数: 1 = NT>1 的核按 bf16 读激活(XB=1), 判据②仍是 f32 单行 ⇒ 过了就是"投机 == 纯解码"在核层面成立 */   /* 第 6 个参数: 0 关; 1 = TMA 批量预取; 2 = 逐线 prefetch 指令(都带 evict_last) */

static uint32_t pick_ksplit(uint32_t out_dim, uint32_t groups, uint32_t nb) {   /* 引擎 v41_q4k_gemv 同式 */
    uint32_t k = 1;
    while (k < WARPS && out_dim * k * groups < 8192u) k <<= 1;
    while (k > 1u && k > nb) k >>= 1;
    return k;
}

static int g_sm = 0;
static bool g_pdl = false;   /* 第 5 个参数: 1 = 编程式依赖发射(引擎走图的样子), 0 = 普通发射(老读数) */
/* 发一次: 普通 <<<>>> 或带 programmaticStreamSerialization 属性(与引擎 ds4_launch_pdl 同一写法) */
template <typename... KArgs, typename... Args>
static void launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t shm, Args... args) {
    if (!g_pdl) { kern<<<grid, block, shm>>>(args...); return; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = shm; cfg.stream = 0;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, kern, args...));
}
/* 发一次. mode: 0 = 引擎规则(组 4~12 KB 走 pipe, 每 SM 4 CTA; 其余 stage); 1 = 一律 pipe, 每 SM 4 CTA; 2 = 一律 pipe, 每 SM 8 CTA;
 * 3 = 一律 stage; 4 = 一律 stage 但 __launch_bounds__ 钉 5 CTA/SM(寄存器 ≤48); 5 = 一律 pipe 5 CTA/SM.
 * ★09-29★: 逐形状表说 pipe 那几个形状(q_b 172 / 共享专家 152 GB/s)与小形状 stage(kv 64 / q_a 175)离墙最远,
 * 而 ksplit 不能动(改了累加序), 能动的只有"谁常驻、驻几个、搬多深、寄存器上限" —— 这些都不改任何一行的加法序。 */
#define NM 6
static const char *g_mname[NM] = { "引擎规则", "pipe×4", "pipe×8", "stage", "stage×5", "pipe×5" };
template <uint32_t NT, uint32_t R, uint32_t XB>
static void launch(float *o, const uint8_t *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t ks, int mode) {
    const uint32_t nb = in_dim / BLK, rpb = WARPS / ks, ngrp = (out_dim + rpb - 1u) / rpb, grp = rpb * nb * BYTES;
    const bool eng_pipe = grp > PIPE_MIN && grp <= PIPE_MAX;
    const bool pipe = mode == 1 || mode == 2 || mode == 5 || (mode == 0 && eng_pipe);
    const dim3 blk(256 / R);
    if (pipe) {
        const uint32_t occ = mode == 2 ? 8u : (mode == 5 ? 5u : 4u);
        uint32_t gx = (uint32_t)g_sm * occ; if (gx > ngrp) gx = ngrp;
        if (mode == 2)      launch_k(k_pipe<NT, R, 8u, XB>, dim3(gx), blk, 2u * grp, o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
        else if (mode == 5) launch_k(k_pipe<NT, R, 5u, XB>, dim3(gx), blk, 2u * grp, o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
        else                launch_k(k_pipe<NT, R, 4u, XB>, dim3(gx), blk, 2u * grp, o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
    } else if (mode == 4) {
        launch_k(k_stage<NT, R, 5u, XB>, dim3(ngrp), blk, grp, o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
    } else {
        launch_k(k_stage<NT, R, 4u, XB>, dim3(ngrp), blk, grp, o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
    }
}

template <uint32_t NT>
static void run_nt(int iters, int only_shape, int only_mode) {
    const shape_t shapes[] = {
        {"q_b 32768x1280", 32768, 1280, 40, 1}, {"wo_a(8组之1) 1024x4096", 1024, 4096, 320, 8},
        {"wo_b 5120x8192", 5120, 8192, 40, 1}, {"q_a 1280x5120", 1280, 5120, 40, 1},
        {"shexp_gate/up 2304x5120", 2304, 5120, 80, 1}, {"shexp_down 5120x2304", 5120, 2304, 40, 1},
        {"kv 512x5120", 512, 5120, 40, 1}, {"idx_q_b 4096x1280", 4096, 1280, 8, 1},
        {"output 129280x5120", 129280, 5120, 1, 1},
        {"wo_a 全8组 8192x4096", 8192, 4096, 40, 1} };   /* 引擎一发 grid.y=8 组连续 19 MB, 与单矩阵 8192 行同一几何(ksplit 1, rpb 8) */
    const char **mname = g_mname;
    double tot[NM] = {0}, best_tot = 0, tot_bytes = 0;
    printf("== NT=%u 行激活(µs / GB/s; %d 种发法, ksplit 不变 ⇒ 全部逐位同; %s发射; 激活 %s)\n", NT, NM, g_pdl ? "PDL 链式" : "普通", g_xb == 2 ? "bf16 块排布(XB=2)" : (g_xb ? "bf16(XB=1)" : "f32"));
    if (g_r != 1) printf("== R=%d: 一个 warp 管 %d 行(CTA 行数 < R 的形状退回 R=1)\n", g_r, g_r);
    int si = -1;
    for (const shape_t &sh : shapes) {
        if (++si != only_shape && only_shape >= 0) continue;
        const uint32_t nb = sh.in_dim / BLK, ks = pick_ksplit(sh.out_dim, sh.groups, nb), rpb = WARPS / ks;
        const uint64_t mbytes = (uint64_t)sh.out_dim * nb * BYTES;
        int ncopy = (int)((256ull << 20) / mbytes) + 1; if (ncopy > 64) ncopy = 64;   /* 轮换多份拷贝压过 L2 */
        uint8_t *h = (uint8_t *)malloc(mbytes);
        uint32_t s = 12345u + sh.out_dim;
        for (uint64_t i = 0; i < mbytes; i++) { s = s * 1664525u + 1013904223u; h[i] = (uint8_t)(s >> 24); }
        for (uint64_t b = 0; b < mbytes / BYTES; b++) {   /* d/dmin 写成正常 f16, 别让随机字节出 NaN/Inf */
            const __half d = __float2half(0.001f + (float)(b % 13) * 1e-4f), m = __float2half(0.002f + (float)(b % 7) * 1e-4f);
            memcpy(h + b * BYTES, &d, 2); memcpy(h + b * BYTES + 2, &m, 2);
        }
        const uint64_t xn = (uint64_t)sh.in_dim * NT, on = (uint64_t)sh.out_dim * NT;
        float *hx = (float *)malloc(xn * 4);
        for (uint64_t i = 0; i < xn; i++) hx[i] = (float)((int)((i * 7 + i / sh.in_dim * 3) % 17) - 8) * 0.0625f;   /* 各行不同 */
        uint8_t **dw = (uint8_t **)malloc(sizeof(uint8_t *) * ncopy);
        for (int c = 0; c < ncopy; c++) { CK(cudaMalloc(&dw[c], mbytes)); CK(cudaMemcpy(dw[c], h, mbytes, cudaMemcpyHostToDevice)); }
        float *dx, *o[NM], *o1; CK(cudaMalloc(&dx, xn * 4)); CK(cudaMemcpy(dx, hx, xn * 4, cudaMemcpyHostToDevice));
        uint16_t *hx16 = (uint16_t *)malloc(xn * 2), *dx16;   /* bf16 副本: hx 的值是 1/16 的整数倍, bf16 精确 */
        for (uint64_t i = 0; i < xn; i++) { uint32_t u; memcpy(&u, &hx[i], 4); hx16[i] = (uint16_t)(u >> 16); }
        CK(cudaMalloc(&dx16, xn * 2)); CK(cudaMemcpy(dx16, hx16, xn * 2, cudaMemcpyHostToDevice));
        uint16_t *hx16p = (uint16_t *)malloc(xn * 2), *dx16p;   /* XB=2 排布: 每块内 lane l 的 [lo4, hi4] 连续 */
        for (uint32_t t = 0; t < NT; t++)
            for (uint32_t b = 0; b < sh.in_dim / BLK; b++)
                for (uint32_t l = 0; l < 32; l++) {
                    const uint32_t gidx = l >> 3, q0 = (l & 7u) * 4u, e_lo = b * BLK + gidx * 64u + q0;
                    for (uint32_t i = 0; i < 4; i++) {
                        hx16p[(uint64_t)t * sh.in_dim + b * BLK + l * 8u + i] = hx16[(uint64_t)t * sh.in_dim + e_lo + i];
                        hx16p[(uint64_t)t * sh.in_dim + b * BLK + l * 8u + 4u + i] = hx16[(uint64_t)t * sh.in_dim + e_lo + 32u + i];
                    }
                }
        CK(cudaMalloc(&dx16p, xn * 2)); CK(cudaMemcpy(dx16p, hx16p, xn * 2, cudaMemcpyHostToDevice));
        const void *xin = g_xb == 2 ? (const void *)dx16p : (g_xb ? (const void *)dx16 : (const void *)dx);
        for (int v = 0; v < NM; v++) CK(cudaMalloc(&o[v], on * 4));
        CK(cudaMalloc(&o1, on * 4));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float us[NM];
        for (int v = 0; v < NM; v++) {
            if (only_mode >= 0 && v != only_mode) { us[v] = 1e9f; continue; }   /* 没跑的发法: 时间记成无穷, 不参与"最快" */
            #define LAUNCH_XR(R_) do { if (g_xb == 2) launch<NT, R_, 2>(o[v], W_, xin, sh.in_dim, sh.out_dim, ks, v); \
                                      else if (g_xb) launch<NT, R_, 1>(o[v], W_, xin, sh.in_dim, sh.out_dim, ks, v); \
                                      else launch<NT, R_, 0>(o[v], W_, xin, sh.in_dim, sh.out_dim, ks, v); } while (0)
            #define LAUNCH_X(O, W) do { const uint8_t *W_ = (W); if (g_r == 2 && (WARPS / ks) >= 2u) LAUNCH_XR(2); else LAUNCH_XR(1); } while (0)
            for (int i = 0; i < 5; i++) LAUNCH_X(o[v], dw[i % ncopy]);
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) LAUNCH_X(o[v], dw[i % ncopy]);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us[v] = ms * 1000.f / iters;
            LAUNCH_X(o[v], dw[0]);
            #undef LAUNCH_X
            #undef LAUNCH_XR
        }
        /* 判据②: 第 t 行单独跑 NT=1(引擎纯解码那条路, 引擎规则) */
        for (uint32_t t = 0; t < NT; t++)
            launch<1, 1, 0>(o1 + (uint64_t)t * sh.out_dim, dw[0], dx + (uint64_t)t * sh.in_dim, sh.in_dim, sh.out_dim, ks, 0);
        CK(cudaDeviceSynchronize());
        float *r[NM], *r1 = (float *)malloc(on * 4);
        for (int v = 0; v < NM; v++) { r[v] = (float *)malloc(on * 4); CK(cudaMemcpy(r[v], o[v], on * 4, cudaMemcpyDeviceToHost)); }
        CK(cudaMemcpy(r1, o1, on * 4, cudaMemcpyDeviceToHost));
        uint64_t bad_m = 0, bad_1 = 0;
        const int ref = only_mode >= 0 ? only_mode : 0;   /* 逐位判据的参照 = 跑过的第一种发法 */
        for (uint64_t i = 0; i < on; i++) {
            for (int v = 0; v < NM; v++) if (v != ref && us[v] < 1e9f && memcmp(&r[ref][i], &r[v][i], 4)) bad_m++;
            if (memcmp(&r[ref][i], &r1[i], 4)) bad_1++;
        }
        int best = ref; for (int v = 0; v < NM; v++) if (us[v] < us[best]) best = v;
        printf("%-24s %6.1f MB ks=%u rpb=%u 组 %5.1f KB |", sh.name, mbytes / 1e6, ks, rpb, (double)rpb * nb * BYTES / 1024.0);
        for (int v = 0; v < NM; v++) { if (us[v] >= 1e9f) continue; printf(" %s %6.1f us %3.0f |", mname[v], us[v], mbytes / (us[v] * 1e3)); }
        printf(" 最快 %s | %s / %s\n", mname[best], bad_m ? "★发法间不同★" : "发法间逐位同", bad_1 ? "★≠单行★" : "== 单行逐位");
        if (g_l2pf && only_mode >= 0) {   /* L2 预取存活试验: 同一发法, 冷读 vs (预取 → 假注意力 → GEMV) */
            float *scr, *sink; const uint64_t sn = (2u << 20) / 4u;
            CK(cudaMalloc(&scr, sn * 4)); CK(cudaMemset(scr, 0, sn * 4)); CK(cudaMalloc(&sink, 4));
            const long long cyc = 242000;   /* ≈100 µs @2.42 GHz, 引擎 n=4 注意力独占段的量级 */
            double t_cold = 0, t_pf = 0, t_pfk = 0, t_dm = 0;
            for (int variant = 0; variant < 2; variant++) {
                for (int i = 0; i < iters + 3; i++) {
                    const uint8_t *wc = dw[i % ncopy];
                    if (variant) { CK(cudaEventRecord(e0)); k_prefetch<<<96, 256>>>(wc, mbytes, g_l2pf); CK(cudaEventRecord(e1)); }
                    if (variant) { CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (i >= 3) t_pfk += ms; }
                    CK(cudaEventRecord(e0)); k_dummy<<<96, 256>>>(scr, sn, sink, cyc); CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1)); { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (i >= 3 && variant) t_dm += ms; }
                    CK(cudaEventRecord(e0)); launch<NT, 1, 0>(o[only_mode], wc, dx, sh.in_dim, sh.out_dim, ks, only_mode); CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1)); { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (i >= 3) { if (variant) t_pf += ms; else t_cold += ms; } }
                }
            }
            CK(cudaGetLastError());
            printf("   L2 预取试验(%s, %s): 假注意力后 GEMV 冷读 %.1f us(%.0f GB/s) → 预取后 %.1f us(%.0f GB/s); 预取核 %.1f us, 假注意力 %.1f us\n",
                   mname[only_mode], g_l2pf == 1 ? "TMA 批量 evict_last" : "逐线 prefetch evict_last",
                   t_cold * 1e3 / iters, mbytes / (t_cold * 1e6 / iters), t_pf * 1e3 / iters, mbytes / (t_pf * 1e6 / iters),
                   t_pfk * 1e3 / iters, t_dm * 1e3 / iters);
            cudaFree(scr); cudaFree(sink);
        }
        for (int v = 0; v < NM; v++) if (us[v] < 1e9f) tot[v] += us[v] * sh.per_step;
        best_tot += us[best] * sh.per_step;
        tot_bytes += (double)mbytes * sh.per_step;
        for (int c = 0; c < ncopy; c++) cudaFree(dw[c]);
        for (int v = 0; v < NM; v++) { cudaFree(o[v]); free(r[v]); }
        free(dw); cudaFree(dx); cudaFree(dx16); free(hx16); cudaFree(dx16p); free(hx16p); cudaFree(o1); free(h); free(hx); free(r1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf("== NT=%u 每步合计(按发射次数加权):", NT);
    for (int v = 0; v < NM; v++) if (tot[v] > 0) printf(" %s %.2f /", mname[v], tot[v] / 1e3);
    printf(" 逐形状取最快 %.2f ms, 字节 %.2f GB ⇒ 墙(242 GB/s) %.2f ms\n", best_tot / 1e3, tot_bytes / 1e9, tot_bytes / 242e9 * 1e3);
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 50;
    const int nt = argc > 2 ? atoi(argv[2]) : 4;
    const int os = argc > 3 ? atoi(argv[3]) : -1, om = argc > 4 ? atoi(argv[4]) : -1;
    g_pdl = argc > 5 && atoi(argv[5]) != 0;
    g_l2pf = argc > 6 ? atoi(argv[6]) : 0;
    g_xb = argc > 7 ? atoi(argv[7]) : 0;
    g_r = argc > 8 ? atoi(argv[8]) : 1;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    g_sm = pr.multiProcessorCount;
    printf("%s SMs=%d L2=%d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    switch (nt) {
        case 1: run_nt<1>(iters, os, om); break; case 2: run_nt<2>(iters, os, om); break; case 3: run_nt<3>(iters, os, om); break;
        case 4: run_nt<4>(iters, os, om); break; case 5: run_nt<5>(iters, os, om); break; case 6: run_nt<6>(iters, os, om); break;
        default: printf("NT 只支持 1..6\n"); return 1;
    }
    return 0;
}
