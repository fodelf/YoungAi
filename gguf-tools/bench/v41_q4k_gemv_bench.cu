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
 * 用法: nvcc -O3 -arch=native -o v41_q4k_gemv_bench v41_q4k_gemv_bench.cu && ./v41_q4k_gemv_bench [iters=50] [NT=4] */
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
template <uint32_t NT, uint32_t R>
__device__ __forceinline__ static void rows_acc(const uint8_t *sw, uint32_t row0, uint32_t nrows_here, uint32_t nblk, uint32_t ksplit,
                                                uint32_t kpart, const float *x, uint32_t x_stride, float (*acc)[NT]) {
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
            const float *xt = x + (uint64_t)t * x_stride;
            const float4 xl = *(const float4 *)(xt + e_lo), xh = *(const float4 *)(xt + e_hi);
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

/* stage: 一个 CTA 一组行(rpb = WARPS/ksplit 行, 连续一段), 整段搬进 shared 再算。blockDim = 32·WARPS/R。 */
template <uint32_t NT, uint32_t R>
__global__ static void __launch_bounds__(256 / R, 4 * R) k_stage(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
                                                                 uint32_t out_dim, uint32_t ksplit, uint32_t x_stride, uint32_t out_stride) {
    extern __shared__ uint4 st[];
    __shared__ float red[WARPS * NT];
    const uint32_t warp = threadIdx.x >> 5, rpb = WARPS / ksplit, nblk = in_dim / BLK;
    const uint32_t kpart = warp % ksplit, row0 = (warp / ksplit) * R;
    const uint32_t r0 = blockIdx.x * rpb, nrows = out_dim - r0 < rpb ? out_dim - r0 : rpb, n16 = nrows * nblk * (BYTES / 16u);
    const uint4 *src = (const uint4 *)(w + (uint64_t)r0 * nblk * BYTES);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) st[i] = __ldcs(src + i);
    __syncthreads();
    float acc[R][NT];
    #pragma unroll
    for (uint32_t j = 0; j < R; j++)
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) acc[j][t] = 0.f;
    rows_acc<NT, R>((const uint8_t *)st, row0, nrows, nblk, ksplit, kpart, x, x_stride, acc);
    rows_finish<NT, R>(acc, out, out_stride, r0, row0, nrows, ksplit, kpart, red);
}
/* pipe: CTA 常驻, 按 rg += gridDim.x 循环行组, cp.async 双缓冲(算第 k 组时搬第 k+1 组) */
template <uint32_t NT, uint32_t R, uint32_t OCC>
__global__ static void __launch_bounds__(256 / R, OCC * R) k_pipe(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
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
        rows_acc<NT, R>((const uint8_t *)(st + (uint64_t)buf * grp16), row0, nrows, nblk, ksplit, kpart, x, x_stride, acc);
        rows_finish<NT, R>(acc, out, out_stride, r0, row0, nrows, ksplit, kpart, red);
        __syncthreads();
    }
}

typedef struct { const char *name; uint32_t out_dim, in_dim, per_step, groups; } shape_t;   /* groups 只用于按引擎口径算 ksplit */

static uint32_t pick_ksplit(uint32_t out_dim, uint32_t groups, uint32_t nb) {   /* 引擎 v41_q4k_gemv 同式 */
    uint32_t k = 1;
    while (k < WARPS && out_dim * k * groups < 8192u) k <<= 1;
    while (k > 1u && k > nb) k >>= 1;
    return k;
}

static int g_sm = 0;
/* 发一次. mode: 0 = 引擎规则(组 4~12 KB 走 pipe, 每 SM 4 CTA; 其余 stage); 1 = 一律 pipe, 每 SM 4 CTA; 2 = 一律 pipe, 每 SM 8 CTA;
 * 3 = 一律 stage. ★09-29★: 逐形状表说 pipe 那几个形状(q_b 172 / 共享专家 152 GB/s)与小形状 stage(kv 64 / q_a 175)离墙最远,
 * 而 ksplit 不能动(改了累加序), 能动的只有"谁常驻、驻几个、搬多深" —— 这些都不改任何一行的加法序。 */
template <uint32_t NT, uint32_t R>
static void launch(float *o, const uint8_t *w, const float *x, uint32_t in_dim, uint32_t out_dim, uint32_t ks, int mode) {
    const uint32_t nb = in_dim / BLK, rpb = WARPS / ks, ngrp = (out_dim + rpb - 1u) / rpb, grp = rpb * nb * BYTES;
    const bool eng_pipe = grp > PIPE_MIN && grp <= PIPE_MAX;
    const bool pipe = mode == 1 || mode == 2 || (mode == 0 && eng_pipe);
    if (pipe) {
        const uint32_t occ = mode == 2 ? 8u : 4u;
        uint32_t gx = (uint32_t)g_sm * occ; if (gx > ngrp) gx = ngrp;
        if (mode == 2) k_pipe<NT, R, 8u><<<gx, 256 / R, 2u * grp>>>(o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
        else           k_pipe<NT, R, 4u><<<gx, 256 / R, 2u * grp>>>(o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
    } else {
        k_stage<NT, R><<<ngrp, 256 / R, grp>>>(o, w, x, in_dim, out_dim, ks, in_dim, out_dim);
    }
}

template <uint32_t NT>
static void run_nt(int iters) {
    const shape_t shapes[] = {
        {"q_b 32768x1280", 32768, 1280, 40, 1}, {"wo_a(8组之1) 1024x4096", 1024, 4096, 320, 8},
        {"wo_b 5120x8192", 5120, 8192, 40, 1}, {"q_a 1280x5120", 1280, 5120, 40, 1},
        {"shexp_gate/up 2304x5120", 2304, 5120, 80, 1}, {"shexp_down 5120x2304", 5120, 2304, 40, 1},
        {"kv 512x5120", 512, 5120, 40, 1}, {"idx_q_b 4096x1280", 4096, 1280, 8, 1},
        {"output 129280x5120", 129280, 5120, 1, 1} };
    const int NM = 4; const char *mname[NM] = { "引擎规则", "pipe×4", "pipe×8", "stage" };
    double tot[NM] = {0, 0, 0, 0}, best_tot = 0, tot_bytes = 0;
    printf("== NT=%u 行激活(µs / GB/s; 四种发法, ksplit 不变 ⇒ 全部逐位同)\n", NT);
    for (const shape_t &sh : shapes) {
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
        for (int v = 0; v < NM; v++) CK(cudaMalloc(&o[v], on * 4));
        CK(cudaMalloc(&o1, on * 4));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float us[NM];
        for (int v = 0; v < NM; v++) {
            for (int i = 0; i < 5; i++) launch<NT, 1>(o[v], dw[i % ncopy], dx, sh.in_dim, sh.out_dim, ks, v);
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) launch<NT, 1>(o[v], dw[i % ncopy], dx, sh.in_dim, sh.out_dim, ks, v);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us[v] = ms * 1000.f / iters;
            launch<NT, 1>(o[v], dw[0], dx, sh.in_dim, sh.out_dim, ks, v);
        }
        /* 判据②: 第 t 行单独跑 NT=1(引擎纯解码那条路, 引擎规则) */
        for (uint32_t t = 0; t < NT; t++)
            launch<1, 1>(o1 + (uint64_t)t * sh.out_dim, dw[0], dx + (uint64_t)t * sh.in_dim, sh.in_dim, sh.out_dim, ks, 0);
        CK(cudaDeviceSynchronize());
        float *r[NM], *r1 = (float *)malloc(on * 4);
        for (int v = 0; v < NM; v++) { r[v] = (float *)malloc(on * 4); CK(cudaMemcpy(r[v], o[v], on * 4, cudaMemcpyDeviceToHost)); }
        CK(cudaMemcpy(r1, o1, on * 4, cudaMemcpyDeviceToHost));
        uint64_t bad_m = 0, bad_1 = 0;
        for (uint64_t i = 0; i < on; i++) { for (int v = 1; v < NM; v++) if (memcmp(&r[0][i], &r[v][i], 4)) bad_m++; if (memcmp(&r[0][i], &r1[i], 4)) bad_1++; }
        int best = 0; for (int v = 1; v < NM; v++) if (us[v] < us[best]) best = v;
        printf("%-24s %6.1f MB ks=%u rpb=%u 组 %5.1f KB |", sh.name, mbytes / 1e6, ks, rpb, (double)rpb * nb * BYTES / 1024.0);
        for (int v = 0; v < NM; v++) printf(" %s %6.1f us %3.0f |", mname[v], us[v], mbytes / (us[v] * 1e3));
        printf(" 最快 %s | %s / %s\n", mname[best], bad_m ? "★发法间不同★" : "发法间逐位同", bad_1 ? "★≠单行★" : "== 单行逐位");
        for (int v = 0; v < NM; v++) tot[v] += us[v] * sh.per_step;
        best_tot += us[best] * sh.per_step;
        tot_bytes += (double)mbytes * sh.per_step;
        for (int c = 0; c < ncopy; c++) cudaFree(dw[c]);
        for (int v = 0; v < NM; v++) { cudaFree(o[v]); free(r[v]); }
        free(dw); cudaFree(dx); cudaFree(o1); free(h); free(hx); free(r1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf("== NT=%u 每步合计(按发射次数加权): 引擎规则 %.2f / pipe×4 %.2f / pipe×8 %.2f / stage %.2f / 逐形状取最快 %.2f ms, 字节 %.2f GB ⇒ 墙(242 GB/s) %.2f ms\n",
           NT, tot[0] / 1e3, tot[1] / 1e3, tot[2] / 1e3, tot[3] / 1e3, best_tot / 1e3, tot_bytes / 1e9, tot_bytes / 242e9 * 1e3);
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 50;
    const int nt = argc > 2 ? atoi(argv[2]) : 4;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    g_sm = pr.multiProcessorCount;
    printf("%s SMs=%d L2=%d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    switch (nt) {
        case 1: run_nt<1>(iters); break; case 2: run_nt<2>(iters); break; case 3: run_nt<3>(iters); break;
        case 4: run_nt<4>(iters); break; case 5: run_nt<5>(iters); break; case 6: run_nt<6>(iters); break;
        default: printf("NT 只支持 1..6\n"); return 1;
    }
    return 0;
}
