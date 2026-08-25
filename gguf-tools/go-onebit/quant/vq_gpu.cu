/* vq_gpu.cu — 量化器 GPU 常驻核(2026-08-18 用户令"改成 gpu 常驻, 先工程再量化")。
 * 两个热点整段上 GPU(段计时归因: kmeans assign + GPTQ 列组反馈 = 层耗时 ~70%):
 *   vqg_assign      k-means 批 assign: 每线程一向量, 码本 shared, 判据与 CPU 同
 *                   (值=|c|²-2v·c, 严格 < 保首现; cblas 浮点序差异容忍, held 对拍验)
 *   vqg_gptq_group  一组 g=128 列的整个"逐 dim 段 assign+误差反馈"循环:
 *                   行间完全独立(反馈只改本行后续列) → 每线程一行, 行内串行 32 段,
 *                   数学与 CPU 行内序逐项一致(Hi double→f32 为唯一近似)。
 * GB10 统一内存: malloc 指针直传, 无显式搬运。每 CPU 线程 thread-local stream。 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

#define VQG_MAX_NC 512
#define VQG_MAX_DIM 8

static __thread cudaStream_t g_vqg_stream = NULL;
static int g_vqg_ok = -1;

extern "C" int vqg_ready(void) {
    if (g_vqg_ok < 0) g_vqg_ok = (cudaFree(0) == cudaSuccess) ? 1 : 0;
    return g_vqg_ok;
}
static cudaStream_t vqg_stream(void) {
    if (!g_vqg_stream) cudaStreamCreateWithFlags(&g_vqg_stream, cudaStreamNonBlocking);
    return g_vqg_stream;
}

/* ---- assign: 每线程一向量 ---- */
__global__ static void vqg_assign_kernel(
        const float *__restrict__ V, int nv,
        const float *__restrict__ C, int nc, int dim, int *__restrict__ idx) {
    extern __shared__ float sh[];              /* C[nc*dim] + c2[nc] */
    float *Cs = sh, *c2 = sh + nc * dim;
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;
    float v[VQG_MAX_DIM];
    for (int d = 0; d < dim; d++) v[d] = V[(size_t)i * dim + d];
    float best = 3.4e38f; int bi = 0;
    for (int c = 0; c < nc; c++) {
        float g = 0.0f;
        for (int d = 0; d < dim; d++) g += v[d] * Cs[c * dim + d];
        float val = c2[c] - 2.0f * g;
        if (val < best) { best = val; bi = c; }   /* 严格 < 保首现, 与 CPU 同判 */
    }
    idx[i] = bi;
}

extern "C" void vqg_assign(const float *V, int nv, const float *C, int nc, int dim, int *idx) {
    cudaStream_t st = vqg_stream();
    size_t shm = (size_t)(nc * dim + nc) * sizeof(float);
    vqg_assign_kernel<<<(nv + 255) / 256, 256, shm, st>>>(V, nv, C, nc, dim, idx);
    cudaStreamSynchronize(st);
}

/* ---- GPTQ 组 v2(08-18 warp 化): 每 warp 一行 —— 32 lane 分摊 512 质心的 assign
 * (16 质心/lane + shfl 归约 argmin, 首现语义=最小 (val,idx) 对), 误差反馈 128 列由
 * 32 lane 并行更新(列间独立)。行内段序与 CPU 一致; Hi 常驻 shared(64KB, GB10 动态上限内)。 */
__global__ static void vqg_gptq_rows_warp_kernel(
        float *__restrict__ Wk, const float *__restrict__ C,
        const float *__restrict__ Hi, int rows, int g, int dim, int nc, int fb,
        int *__restrict__ sidx) {
    extern __shared__ float sh[];
    float *Cs = sh, *c2 = sh + nc * dim;          /* 码本 + |c|² */
    float *His = c2 + nc;                          /* [g,g] f32 (fb 时装载) */
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    if (fb) for (int i = threadIdx.x; i < g * g; i += blockDim.x) His[i] = Hi[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nseg = g / dim;
    for (int r = blockIdx.x * nwarp + warp; r < rows; r += gridDim.x * nwarp) {
        float *wkr = Wk + (size_t)r * g;
        for (int s = 0; s < nseg; s++) {
            const int jj = s * dim;
            float seg[VQG_MAX_DIM];
            for (int d = 0; d < dim; d++) seg[d] = wkr[jj + d];
            /* lane 分质心: 每 lane nc/32 个, 保序 argmin(先 val 后 idx = 全局首现) */
            float best = 3.4e38f; int bi = nc;
            for (int c = lane; c < nc; c += 32) {
                float gd = 0.0f;
                for (int d = 0; d < dim; d++) gd += seg[d] * Cs[c * dim + d];
                float val = c2[c] - 2.0f * gd;
                if (val < best || (val == best && c < bi)) { best = val; bi = c; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float ob = __shfl_down_sync(0xffffffffu, best, off);
                int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ob < best || (ob == best && oi < bi)) { best = ob; bi = oi; }
            }
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (lane == 0) sidx[(size_t)r * nseg + s] = bi;
            if (fb && jj + dim < g) {
                const float *q = Cs + bi * dim;
                for (int d = 0; d < dim; d++) {
                    float hjj = His[(size_t)(jj + d) * g + jj + d];
                    if (fabsf(hjj) < 1e-30f) hjj = 1e-30f;
                    float er = (seg[d] - q[d]) / hjj;
                    if (er == 0.0f) continue;
                    const float *hrow = His + (size_t)(jj + d) * g;
                    for (int j2 = jj + dim + lane; j2 < g; j2 += 32)
                        wkr[j2] -= er * hrow[j2];   /* 列间独立, lane 并行 */
                }
                __syncwarp();
            }
        }
    }
}

/* ---- GPTQ 组: 每线程一行, 行内串行段循环(与 CPU 逐项同序) ---- */
__global__ static void vqg_gptq_rows_kernel(
        float *__restrict__ Wk,                /* [rows, g] in/out */
        const float *__restrict__ C,           /* [nc, dim] */
        const float *__restrict__ Hi,          /* [g, g] f32(double 转换) */
        int rows, int g, int dim, int nc, int fb,
        int *__restrict__ sidx) {              /* [rows, g/dim] */
    extern __shared__ float sh[];
    float *Cs = sh, *c2 = sh + nc * dim;
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    __syncthreads();
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float *wkr = Wk + (size_t)r * g;
    const int nseg = g / dim;
    for (int s = 0; s < nseg; s++) {
        const int jj = s * dim;
        float seg[VQG_MAX_DIM];
        for (int d = 0; d < dim; d++) seg[d] = wkr[jj + d];
        float best = 3.4e38f; int bi = 0;
        for (int c = 0; c < nc; c++) {
            float gd = 0.0f;
            for (int d = 0; d < dim; d++) gd += seg[d] * Cs[c * dim + d];
            float val = c2[c] - 2.0f * gd;
            if (val < best) { best = val; bi = c; }
        }
        sidx[(size_t)r * nseg + s] = bi;
        if (fb && jj + dim < g) {
            const float *q = Cs + bi * dim;
            for (int d = 0; d < dim; d++) {
                float hjj = Hi[(size_t)(jj + d) * g + jj + d];
                if (fabsf(hjj) < 1e-30f) hjj = 1e-30f;
                float er = (seg[d] - q[d]) / hjj;
                if (er == 0.0f) continue;
                const float *hrow = Hi + (size_t)(jj + d) * g;
                for (int j2 = jj + dim; j2 < g; j2++) wkr[j2] -= er * hrow[j2];
            }
        }
    }
}

/* ---- v3 全矩阵批量(08-18): 组间独立 → 32 组一次 launch(grid.y=组), 调用数
 * 24.6k→768/层, 消灭 per-组 sync(实测 7.8ms/调用 = 193s/层的本体)。
 * 每 warp 一行×一组: 行内段循环与 CPU 同序; Wk 列窗口驻全局 buffer。 */
__global__ static void vqg_gptq_full_kernel(
        float *__restrict__ Wk,               /* [rows, cols] 工作副本(in/out) */
        const float *__restrict__ C,
        const float *__restrict__ Hi_all,     /* [ngrp, g, g] f32; NULL=无反馈 */
        int rows, int cols, int grp, int dim, int nc,
        int *__restrict__ sidx) {             /* [rows, cols/dim] */
    extern __shared__ float sh[];
    float *Cs = sh, *c2 = sh + nc * dim;
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) Cs[i] = C[i];
    __syncthreads();
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += Cs[c * dim + d] * Cs[c * dim + d];
        c2[c] = s;
    }
    __syncthreads();
    const int grpi = blockIdx.y;
    const int j0 = grpi * grp;
    const int g = (cols - j0 < grp) ? (cols - j0) : grp;
    const float *Hi = Hi_all ? Hi_all + (size_t)grpi * grp * grp : NULL;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nseg_all = cols / dim;
    const int nseg = g / dim;
    for (int r = blockIdx.x * nwarp + warp; r < rows; r += gridDim.x * nwarp) {
        float *wkr = Wk + (size_t)r * cols + j0;
        for (int s = 0; s < nseg; s++) {
            const int jj = s * dim;
            float seg[VQG_MAX_DIM];
            for (int d = 0; d < dim; d++) seg[d] = wkr[jj + d];
            float best = 3.4e38f; int bi = nc;
            for (int c = lane; c < nc; c += 32) {
                float gd = 0.0f;
                for (int d = 0; d < dim; d++) gd += seg[d] * Cs[c * dim + d];
                float val = c2[c] - 2.0f * gd;
                if (val < best || (val == best && c < bi)) { best = val; bi = c; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float ob = __shfl_down_sync(0xffffffffu, best, off);
                int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (ob < best || (ob == best && oi < bi)) { best = ob; bi = oi; }
            }
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (lane == 0) sidx[(size_t)r * nseg_all + (j0 + jj) / dim] = bi;
            if (Hi && jj + dim < g) {
                const float *q = Cs + bi * dim;
                for (int d = 0; d < dim; d++) {
                    float hjj = Hi[(size_t)(jj + d) * g + jj + d];
                    if (fabsf(hjj) < 1e-30f) hjj = 1e-30f;
                    float er = (seg[d] - q[d]) / hjj;
                    if (er == 0.0f) continue;
                    const float *hrow = Hi + (size_t)(jj + d) * g;
                    for (int j2 = jj + dim + lane; j2 < g; j2 += 32)
                        wkr[j2] -= er * hrow[j2];
                }
                __syncwarp();
            }
        }
    }
}

extern "C" void vqg_gptq_full(float *Wk, const float *C, const double *Hi_all_d,
                              int rows, int cols, int grp, int dim, int nc,
                              int has_fb, int *sidx) {
    static __thread float *hi_f = NULL;
    static __thread int hi_cap = 0;
    const int ngrp = (cols + grp - 1) / grp;
    if (has_fb) {
        const int need = ngrp * grp * grp;
        if (hi_cap < need) { if (hi_f) cudaFreeHost(hi_f); cudaMallocHost((void **)&hi_f, (size_t)need * sizeof(float)); hi_cap = need; }
        for (int i = 0; i < need; i++) hi_f[i] = (float)Hi_all_d[i];
    }
    cudaStream_t st = vqg_stream();
    size_t shm = (size_t)(nc * dim + nc) * sizeof(float);
    dim3 grid((rows + 3) / 4, ngrp);
    vqg_gptq_full_kernel<<<grid, 128, shm, st>>>(
        Wk, C, has_fb ? hi_f : NULL, rows, cols, grp, dim, nc, sidx);
    cudaStreamSynchronize(st);
}

extern "C" void vqg_gptq_group(float *Wk, const float *C, const double *Hi_d,
                               int rows, int g, int dim, int nc, int has_fb, int *sidx) {
    static __thread float *hi_f = NULL;
    static __thread int hi_cap = 0;
    if (has_fb) {
        if (hi_cap < g * g) { if (hi_f) cudaFreeHost(hi_f); cudaMallocHost((void **)&hi_f, (size_t)g * g * sizeof(float)); hi_cap = g * g; }
        for (int i = 0; i < g * g; i++) hi_f[i] = (float)Hi_d[i];
    }
    cudaStream_t st = vqg_stream();
    size_t shm2 = (size_t)(nc * dim + nc + (has_fb ? g * g : 0)) * sizeof(float);
    if (shm2 <= 96 * 1024) {   /* warp 版: Hi 进 shared(128²×4=64KB, GB10 上限内) */
        static int attr_set = 0;
        if (!attr_set) { cudaFuncSetAttribute(vqg_gptq_rows_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); attr_set = 1; }
        vqg_gptq_rows_warp_kernel<<<(rows + 3) / 4, 128, shm2, st>>>(
            Wk, C, has_fb ? hi_f : NULL, rows, g, dim, nc, has_fb, sidx);
    } else {
        size_t shm = (size_t)(nc * dim + nc) * sizeof(float);
        vqg_gptq_rows_kernel<<<(rows + 127) / 128, 128, shm, st>>>(
            Wk, C, has_fb ? hi_f : NULL, rows, g, dim, nc, has_fb, sidx);
    }
    cudaStreamSynchronize(st);
}

/* ---- B 回放 dequant(08-18 第二刀): 位流→fp32 权重整矩阵, 每线程一行 ----
 * 语义与 vq_qc.h vq_unpack_dequant 逐位同(LE 位流+fp16 码本/行gain);
 * B 回放评估 18s/层的主体=768 矩阵 CPU 标量位流解码。 */
__global__ static void vqg_dequant_kernel(
        const uint8_t *__restrict__ cb_h,     /* fp16 码本 [nc*dim] */
        const uint8_t *__restrict__ gr_h,     /* fp16 行 gain [rows] */
        const uint8_t *__restrict__ ix,       /* 位流 */
        float *__restrict__ W, int rows, int cols, int dim, int nc, int nbit) {
    extern __shared__ float sh[];
    float *Cs = sh;                            /* nc*dim 码本 fp32 */
    for (int i = threadIdx.x; i < nc * dim; i += blockDim.x) {
        uint16_t h = (uint16_t)cb_h[i * 2] | ((uint16_t)cb_h[i * 2 + 1] << 8);
        Cs[i] = __half2float(*(const __half *)&h);
    }
    __syncthreads();
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    uint16_t gh = (uint16_t)gr_h[r * 2] | ((uint16_t)gr_h[r * 2 + 1] << 8);
    const float g = __half2float(*(const __half *)&gh);
    const int nidx_row = cols / dim;
    const size_t i0 = (size_t)r * nidx_row;
    float *wr = W + (size_t)r * cols;
    for (int c = 0; c < nidx_row; c++) {
        uint32_t v;
        if (nbit == 8) v = ix[i0 + c];
        else {
            const size_t bit = (i0 + c) * (size_t)nbit;
            uint32_t w0 = ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16);
            v = (w0 >> (bit & 7)) & ((1u << nbit) - 1u);
        }
        const float *ce = Cs + v * dim;
        for (int d = 0; d < dim; d++) wr[c * dim + d] = g * ce[d];
    }
}

extern "C" int vqg_unpack_dequant(const uint8_t *pay, float *W,
                                  int rows, int cols, int dim, int nc, int nbit) {
    if (nc > VQG_MAX_NC * 2 || dim > VQG_MAX_DIM) return 0;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    cudaStream_t st = vqg_stream();
    size_t shm = (size_t)nc * dim * sizeof(float);
    vqg_dequant_kernel<<<(rows + 127) / 128, 128, shm, st>>>(cb, gr, ix, W, rows, cols, dim, nc, nbit);
    return cudaStreamSynchronize(st) == cudaSuccess;
}

/* ---- 批量 dequant(08-18 第三刀): 一层 768 矩阵一次 launch, 消 99k 次 sync ----
 * B 回放段计时实锤: dequant 752s 累计 vs 前向 83s, 7.6ms/调用全是 launch+sync 争用。 */
typedef struct { uint64_t pay_off; uint64_t dst_off; int rows, cols, nc, nbit; } vqg_deq_job;

__global__ static void vqg_dequant_batch_kernel(
        const uint8_t *__restrict__ base, float *__restrict__ dst_base,
        const vqg_deq_job *__restrict__ jobs, int njobs, int dim) {
    const int ji = blockIdx.y;
    if (ji >= njobs) return;
    const vqg_deq_job j = jobs[ji];
    const uint8_t *pay = base + j.pay_off;
    const uint8_t *cb_h = pay + 16;
    const uint8_t *gr_h = cb_h + (size_t)j.nc * dim * 2;
    const uint8_t *ix = gr_h + (size_t)j.rows * 2;
    float *W = dst_base + j.dst_off;
    extern __shared__ float sh[];
    float *Cs = sh;
    for (int i = threadIdx.x; i < j.nc * dim; i += blockDim.x) {
        uint16_t h = (uint16_t)cb_h[i * 2] | ((uint16_t)cb_h[i * 2 + 1] << 8);
        Cs[i] = __half2float(*(const __half *)&h);
    }
    __syncthreads();
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < j.rows; r += gridDim.x * blockDim.x) {
        uint16_t gh = (uint16_t)gr_h[r * 2] | ((uint16_t)gr_h[r * 2 + 1] << 8);
        const float g = __half2float(*(const __half *)&gh);
        const int nidx_row = j.cols / dim;
        const size_t i0 = (size_t)r * nidx_row;
        float *wr = W + (size_t)r * j.cols;
        for (int c = 0; c < nidx_row; c++) {
            uint32_t v;
            if (j.nbit == 8) v = ix[i0 + c];
            else {
                const size_t bit = (i0 + c) * (size_t)j.nbit;
                uint32_t w0 = ix[bit >> 3] | ((uint32_t)ix[(bit >> 3) + 1] << 8) | ((uint32_t)ix[(bit >> 3) + 2] << 16);
                v = (w0 >> (bit & 7)) & ((1u << j.nbit) - 1u);
            }
            const float *ce = Cs + v * dim;
            for (int d = 0; d < dim; d++) wr[c * dim + d] = g * ce[d];
        }
    }
}

extern "C" int vqg_dequant_batch(const uint8_t *base, float *dst_base,
                                 const vqg_deq_job *jobs_h, int njobs, int dim, int nc_max) {
    static __thread vqg_deq_job *jobs_d = NULL;
    static __thread int jcap = 0;
    if (jcap < njobs) { if (jobs_d) cudaFree(jobs_d); cudaMalloc((void **)&jobs_d, (size_t)njobs * sizeof(vqg_deq_job)); jcap = njobs; }
    cudaStream_t st = vqg_stream();
    cudaMemcpyAsync(jobs_d, jobs_h, (size_t)njobs * sizeof(vqg_deq_job), cudaMemcpyHostToDevice, st);
    size_t shm = (size_t)nc_max * dim * sizeof(float);
    dim3 grid(16, njobs);
    vqg_dequant_batch_kernel<<<grid, 128, shm, st>>>(base, dst_base, jobs_d, njobs, dim);
    return cudaStreamSynchronize(st) == cudaSuccess;
}

extern "C" int vqg_alloc_managed(void **p, size_t bytes) {
    if (cudaMallocManaged(p, bytes) != cudaSuccess) return 0;
    return 1;
}
