/* v41_gr_solve.cu — 权重侧逐专家反修: 重解每个专家 down 矩阵的【逐输出通道增益】(2026-09-13)。
 *
 * 【为什么是这个形态】09-13 的针 0 判死了"每层 MoE 出口加 y += x·(B·A)"这一族: 它对出口分布的
 * 一阶效应 ≈ 0(ΔKLD 对步长 β 是纯二次, 拟合 a=−0.016 b=0.507, 最优步长 β*=0.016 收益 0.0001,
 * 比判决器分辨率小两个数量级)。原因是那个修正是 x 的线性函数, 而出口在乎的方向不在 x 的线性空间里。
 * 这里换成改权重本身: 量化文件里每个专家的每个矩阵存的是 W[r][c] = 码本[idx][d] · g[r](见 vq_fmt.h),
 * g 是逐行标量。down 矩阵的"行"就是输出通道 ⇒ 重解 g 等于给每个专家的每个输出通道一个新增益。
 * 这样注进去的修正 Δ 随"选中了哪几个专家、隐层是什么"逐 token 变化 —— 不是 x 的线性函数。
 *
 * 【数学】引擎这一层出的是 y[i][d] = round_bf16( Σ_k rw[i][k]·ye[i][k][d] + ysh[i][d] )(逐位, 取料自检验过),
 * ye 是逐专家 down 输出(未乘路由权重), ysh 是 shared 专家输出。把专家 e 的增益整体缩 s_e[d] 倍:
 *     ŷ[i][d] = ysh[i][d] + Σ_k s_{e(i,k)}[d] · u[i][k][d],   u[i][k][d] = rw[i][k]·ye[i][k][d]
 * 靶是 FP 的整块输出 y_fp。**逐输出通道 d 完全独立**(增益是对角的, 不混通道), 每个 d 上是一个
 * 384 维最小二乘, 方程数 = 拟合行数。加岭把解锚在"不改"(s=1)上:
 *     min_s  Σ_i ( ysh + Σ_k s·u − y_fp )²  +  λ · den_bar[d] · Σ_e (s_e − 1)²
 * ★den_bar[d] = 本通道上所有专家能量的平均, 是【全体共用的一个尺度】★。
 * 实撞(09-13 L00): 第一版拿每个专家自己的能量 den_e 当尺度, 即 s_e = (num/den_e + λ)/(1 + λ) ——
 * 那个式子对 s 的绝对偏离**根本没有约束**: 冷门专家在拟合段只被选中几次, den_e 很小, 岭跟着一起小,
 * 解被少数几行推到极端值; train +32.7% 而 val −2685%, 且 λ 从 0.01 加到 1 只把 val 从 −2686 拉到 −1002
 * (岭在"变大"但相对强度没变)。用统一尺度后, 冷门专家会被强拉回 1, 热门专家才动得了。
 * 路由有多不均是这条路的核心风险: 384 个专家里被选中次数最少的那些, 本来就没有数据支撑改它。
 *
 * 【解法】Gauss-Seidel 逐专家轮转: 每个专家只碰它名下的行(平均 8192×6/384 ≈ 128 行), 解完立刻把
 * 残差更新掉, 下一个专家看到的就是最新状态。3 轮足够(专家之间的耦合只来自同一 token 里的 6 个 pick,
 * 共现很稀疏)。不解正规方程组: 5120 个 384×384 要 3 GB 且大部分是零。
 *
 * 【体积与带宽】产物 = 每层 [n_expert][D] f16。40 层 × 384 × 5120 × 2 B = 157 MB(+0.0023 bpw)。
 * **解码带宽零增量** —— 引擎本来每 token 就要读这些行增益(它在 VQ 载荷里), 只是改读覆盖表。
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GRCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "★CUDA %s:%d %s★\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)

/* resid[i][d] = y_fp[i][d] − ysh[i][d] − Σ_k s[e(i,k)][d]·u[i][k][d]。s 全 1 时就是"不改"的残差。 */
__global__ static void gr_resid_kernel(float *resid, const float *yfp, const float *ysh, const float *ye,
                                       const float *rw, const int *sel, const float *s,
                                       int n, int nu, int D) {
    const int i = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = ysh[(size_t)i * D + d];
    for (int k = 0; k < nu; k++) {
        const int e = sel[(size_t)i * nu + k];
        if (e < 0) continue;
        acc += s[(size_t)e * D + d] * rw[(size_t)i * nu + k] * ye[((size_t)i * nu + k) * D + d];
    }
    resid[(size_t)i * D + d] = yfp[(size_t)i * D + d] - acc;
}

/* 一个专家一发: 每线程管一个输出通道 d。先把该专家当前的贡献加回残差, 解一维带岭最小二乘, 再把新贡献减掉。
 * rows/nrow 是该专家名下的 (i,k) 对(只含拟合行 —— val 行不许参与解算)。 */
__global__ static void gr_expert_kernel(float *s, float *resid, const float *ye, const float *rw,
                                        const int *rows, int nrow, int e, float lam, int nu, int D,
                                        const float *den_bar) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float s_old = s[(size_t)e * D + d];
    float num = 0.f, den = 0.f;
    for (int j = 0; j < nrow; j++) {
        const int pk = rows[j], i = pk / nu, k = pk - i * nu;
        const float u = rw[(size_t)pk] * ye[(size_t)pk * D + d];
        const float r = resid[(size_t)i * D + d] + s_old * u;   /* 加回本专家的贡献 */
        num += u * r; den += u * u;
        (void)k;
    }
    const float ridge = lam * den_bar[d];                       /* 岭锚在 s=1, 尺度是本通道的【平均】专家能量 */
    const float s_new = den > 0.f ? (num + ridge) / (den + ridge) : 1.f;
    s[(size_t)e * D + d] = s_new;
    if (s_new != s_old) {
        const float ds = s_new - s_old;
        for (int j = 0; j < nrow; j++) {
            const int pk = rows[j], i = pk / nu;
            resid[(size_t)i * D + d] -= ds * rw[(size_t)pk] * ye[(size_t)pk * D + d];
        }
    }
}

/* den_bar[d] = (Σ_e Σ_{i∋e} u²) / n_expert, 只用拟合行。一个 block 一个通道, 线程分摊所有拟合对。 */
__global__ static void gr_denbar_kernel(float *den_bar, const float *ye, const float *rw, const int *sel,
                                        int nfitpair, int n_expert, int D) {
    __shared__ double sh[256];
    const int d = blockIdx.x;
    double acc = 0.0;
    for (int p = threadIdx.x; p < nfitpair; p += blockDim.x) {
        if (sel[p] < 0) continue;
        const double u = (double)rw[p] * ye[(size_t)p * D + d];
        acc += u * u;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) den_bar[d] = (float)(sh[0] / (double)n_expert);
}

/* Σ resid² 与 Σ 靶²(靶 = y_fp − ysh − Σ_k u, 即 s≡1 时的残差)分行段统计, 用来报 train/val 的挽回率。 */
__global__ static void gr_norm_kernel(double *out, const float *resid, int row0, int nrow, int D) {
    __shared__ double sh[256];
    double acc = 0.0;
    for (long long t = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
         t < (long long)nrow * D; t += (long long)blockDim.x * gridDim.x) {
        const double v = resid[(size_t)row0 * D + t];
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

/* 解一层。输入全是设备指针(dye/drw/dsel/dyfp/dysh 按【置换后行序】: 拟合行在前 nfit 行)。
 * ds[n_expert][D] 输出缩放因子(锚 1)。out_train/out_val = 1 − ‖残差‖/‖靶‖ 的百分比(与现役放大器同口径,
 * 幅值比不是平方能量比)。返回 0 成功。 */
extern "C" int v41_gr_solve_layer_gpu(const float *dye, const float *drw, const int *dsel,
                                      const float *dyfp, const float *dysh,
                                      int n, int nfit, int nu, int D, int n_expert,
                                      float lam, int niter, float *ds,
                                      float *out_train, float *out_val) {
    const size_t nD = (size_t)n * D, npair = (size_t)n * nu;
    float *resid = NULL, *dbar = NULL; double *dsum = NULL;
    GRCK(cudaMalloc((void **)&resid, nD * 4));
    GRCK(cudaMalloc((void **)&dbar, (size_t)D * 4));
    GRCK(cudaMalloc((void **)&dsum, sizeof(double)));

    /* 每个专家的拟合行 (i,k) 对索引表: 主机建, 一次上设备。val 行绝不进表 —— 解算只许看拟合行。 */
    int *hsel = (int *)malloc(npair * 4);
    if (!hsel) { cudaFree(resid); cudaFree(dsum); return -1; }
    GRCK(cudaMemcpy(hsel, dsel, npair * 4, cudaMemcpyDeviceToHost));
    int *cnt = (int *)calloc((size_t)n_expert, 4);
    const size_t nfitpair = (size_t)nfit * nu;
    for (size_t p = 0; p < nfitpair; p++) if (hsel[p] >= 0 && hsel[p] < n_expert) cnt[hsel[p]]++;
    int *off = (int *)malloc(((size_t)n_expert + 1) * 4);
    off[0] = 0;
    for (int e = 0; e < n_expert; e++) off[e + 1] = off[e] + cnt[e];
    int *rows = (int *)malloc((size_t)off[n_expert] * 4);
    int *cur = (int *)calloc((size_t)n_expert, 4);
    for (size_t p = 0; p < nfitpair; p++) {
        const int e = hsel[p];
        if (e >= 0 && e < n_expert) rows[off[e] + cur[e]++] = (int)p;
    }
    int *drows = NULL;
    GRCK(cudaMalloc((void **)&drows, (size_t)(off[n_expert] > 0 ? off[n_expert] : 1) * 4));
    if (off[n_expert] > 0) GRCK(cudaMemcpy(drows, rows, (size_t)off[n_expert] * 4, cudaMemcpyHostToDevice));

    gr_denbar_kernel<<<D, 256>>>(dbar, dye, drw, dsel, (int)nfitpair, n_expert, D);
    GRCK(cudaGetLastError());
    const dim3 blk(256), grd((D + 255) / 256, n);
    /* ① s ≡ 1 的残差 = 靶本身, 量下来当分母 */
    GRCK(cudaMemset(ds, 0, (size_t)n_expert * D * 4));
    {   float *ones = (float *)malloc((size_t)n_expert * D * 4);
        for (size_t t = 0; t < (size_t)n_expert * D; t++) ones[t] = 1.f;
        GRCK(cudaMemcpy(ds, ones, (size_t)n_expert * D * 4, cudaMemcpyHostToDevice));
        free(ones);
    }
    gr_resid_kernel<<<grd, blk>>>(resid, dyfp, dysh, dye, drw, dsel, ds, n, nu, D);
    GRCK(cudaGetLastError());
    double tgt_fit = 0, tgt_val = 0, res_fit = 0, res_val = 0;
    const int nval = n - nfit;
    GRCK(cudaMemset(dsum, 0, sizeof(double)));
    gr_norm_kernel<<<256, 256>>>(dsum, resid, 0, nfit, D);
    GRCK(cudaMemcpy(&tgt_fit, dsum, sizeof(double), cudaMemcpyDeviceToHost));
    if (nval > 0) {
        GRCK(cudaMemset(dsum, 0, sizeof(double)));
        gr_norm_kernel<<<256, 256>>>(dsum, resid, nfit, nval, D);
        GRCK(cudaMemcpy(&tgt_val, dsum, sizeof(double), cudaMemcpyDeviceToHost));
    }

    /* ② Gauss-Seidel 轮转 */
    for (int it = 0; it < niter; it++) {
        for (int e = 0; e < n_expert; e++) {
            const int nrow = off[e + 1] - off[e];
            if (nrow <= 0) continue;
            gr_expert_kernel<<<(D + 255) / 256, 256>>>(ds, resid, dye, drw, drows + off[e], nrow, e, lam, nu, D, dbar);
        }
        GRCK(cudaGetLastError());
        GRCK(cudaDeviceSynchronize());
    }

    /* ③ val 行的残差要重算 —— Gauss-Seidel 只更新了拟合行的残差(它只走拟合行的索引表) */
    gr_resid_kernel<<<grd, blk>>>(resid, dyfp, dysh, dye, drw, dsel, ds, n, nu, D);
    GRCK(cudaGetLastError());
    GRCK(cudaMemset(dsum, 0, sizeof(double)));
    gr_norm_kernel<<<256, 256>>>(dsum, resid, 0, nfit, D);
    GRCK(cudaMemcpy(&res_fit, dsum, sizeof(double), cudaMemcpyDeviceToHost));
    if (nval > 0) {
        GRCK(cudaMemset(dsum, 0, sizeof(double)));
        gr_norm_kernel<<<256, 256>>>(dsum, resid, nfit, nval, D);
        GRCK(cudaMemcpy(&res_val, dsum, sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (out_train) *out_train = tgt_fit > 0 ? (float)((1.0 - sqrt(res_fit / tgt_fit)) * 100.0) : 0.f;
    if (out_val) *out_val = tgt_val > 0 ? (float)((1.0 - sqrt(res_val / tgt_val)) * 100.0) : 0.f;

    free(hsel); free(cnt); free(off); free(rows); free(cur);
    cudaFree(drows); cudaFree(resid); cudaFree(dbar); cudaFree(dsum);
    return 0;
}
