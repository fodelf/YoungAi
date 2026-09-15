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
 * 【体积与带宽】产物 = 每层 [n_expert][D] 的 fp4x32 码字(4.25 bpw)。37 层 × 1.04 MB = 38.6 MB
 * (+0.00035 bpw)。**解码带宽零增量** —— 引擎本来每 token 就要读这些行增益(它在 VQ 载荷里),
 * 只是改读覆盖表。
 *
 * 【★格点直解: 解出来的就是 fp4, 不是解完再压★】(2026-09-14, 用户令"反修输出直接就是 fp4,
 * 而不是转换数据")
 * 落地位宽是 4 bit, 而 E2M1 只有 8 个幅值格点(±{0,.5,1,1.5,2,3,4,6}×每 32 个数一个 2 的幂块缩放),
 * 相对精度就是 10% 量级 —— 这是位宽的物理精度, 不是编码器没调好(块缩放已逐块搜 6 档)。
 * 所以"先用 f32 自由解, 落盘时再压成 fp4"必然要白扔掉 11% 的修正量(09-14 实测: 现役 f32 产物
 * 重编码, 相对修正量的 RMS 误差 11.18%), 而且那份误差没有任何人接管。
 *
 * 这里改成: **Gauss-Seidel 每解完一个专家, 立刻把它的 s 投影到 fp4 格点, 用投影后的值更新残差**。
 * 于是 ①这个专家的量化误差马上变成残差的一部分, ②排在后面的专家在解自己的通道时会看见它并补偿,
 * ③第 2、3 轮轮转里每个专家都会再看一次别人的量化误差。最终 s 完全住在格点上 ⇒ 落盘零转换损失,
 * 报出来的 train/val 天然就是【落地态】的读数(以前是解算态择 λ、落地态部署, 两个数不是一回事)。
 * 与 GPTQ/OBQ 的误差补偿是同一个道理, 只是这里的"未知数"是逐专家逐通道的一维增益。
 *
 * ★码字由 GPU 直接生成, CPU 侧只负责【解码自证】★: 落盘的是这里输出的 dpk 字节, 主机拿
 * src/common 的唯一解码器 ds4_deq_fp4x32 解回来, 必须与设备上的 s 逐位相等, 不等就停车 ——
 * 这样"GPU 编码"与"全仓唯一格式基元"之间不会悄悄漂开(舍入与饱和都走 ds4_fp8.h 的同一段代码)。
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../src/common/ds4_fp8.h"   /* e2m1 舍入 / e8m0 / nibble 表: host 与 device 同一份实现 */

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

/* 一个专家一发的第①步: 每线程管一个输出通道 d, 解一维带岭最小二乘。
 * ★只写 s, 不动残差★ —— 残差要等 s 被投影到 fp4 格点之后再按【格点值】更新(第③步),
 * 否则残差记的是一个盘上根本不存在的 s, 后面的专家补偿的就是假误差。
 * s_old 存一份出来给第③步用(它要的是"这一轮实际挪了多少")。
 * rows/nrow 是该专家名下的 (i,k) 对(只含拟合行 —— val 行不许参与解算)。 */
__global__ static void gr_expert_kernel(float *s, float *s_old_out, const float *resid,
                                        const float *ye, const float *rw,
                                        const int *rows, int nrow, int e, float lam, int nu, int D,
                                        const float *den_bar) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float s_old = s[(size_t)e * D + d];
    s_old_out[d] = s_old;
    float num = 0.f, den = 0.f;
    for (int j = 0; j < nrow; j++) {
        const int pk = rows[j], i = pk / nu;
        const float u = rw[(size_t)pk] * ye[(size_t)pk * D + d];
        const float r = resid[(size_t)i * D + d] + s_old * u;   /* 加回本专家的贡献 */
        num += u * r; den += u * u;
    }
    const float ridge = lam * den_bar[d];                       /* 岭锚在 s=1, 尺度是本通道的【平均】专家能量 */
    s[(size_t)e * D + d] = den > 0.f ? (num + ridge) / (den + ridge) : 1.f;
}

/* 第②步 ★把这个专家的增益投影到 fp4 格点, 并直接产出落盘码字★。
 * 一个 block = 一个 32 通道块(= 盘上一个 fp4x32 块, 17 B), 32 个线程各管一个通道。
 *
 * 【存的是 s−1 不是 s】格点是 ±{0,.5,1,1.5,2,3,4,6}×块缩放, 而增益恒在 1 附近(|s−1| 中位 0.09~0.31)。
 * 直接存 s, 块缩放被最大值(≈1.3)定死, 格点间隔 0.11 跟修正量本身同量级 ⇒ 实测失真 60.5%, 等于
 * 把插件抹掉大半。存 s−1 以 0 为中心, 块缩放贴着修正幅度走, 失真 11% 量级。
 * 【块缩放逐块搜 6 档】e0 = ceil(log2(amax/6)) 是"一个都不裁"的最小指数; 从 e0−1(允许少量裁剪,
 * 重尾块上反而更准)到 e0+4 各试一次, 取块内平方误差最小的 —— 与主机 ds4_quant_fp4x32 逐式同。
 * 【每次都写码字】对已经在格点上的值再投影一次未必选出同一个块缩放(amax 变了), 所以不赌"幂等":
 * 每次投影都把码字写出来, 落盘用的永远是最后一次投影的那一份, 与设备上的 s 严格对应。 */
__global__ static void gr_fp4_quant_kernel(float *s, uint8_t *pk, int e, int D) {
    __shared__ float sh[32], v[32];
    __shared__ uint8_t nb[32];
    const int j = threadIdx.x, b = blockIdx.x;
    const size_t base = (size_t)e * D + (size_t)b * 32;
    uint8_t *blk = pk + ((size_t)e * (D / 32) + b) * 17;
    v[j] = s[base + j] - 1.0f;
    sh[j] = fabsf(v[j]);
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) { if (j < o && sh[j + o] > sh[j]) sh[j] = sh[j + o]; __syncthreads(); }
    const float amax = sh[0];
    __syncthreads();
    if (!(amax > 0.0f)) {   /* 全零块: 缩放存 127(=2^0), nibble 全 0 —— 位型唯一, 不留随机残字节 */
        if (j < 16) blk[j] = 0;
        if (j == 0) blk[16] = 127;
        s[base + j] = 1.0f;
        return;
    }
    const int e0 = (int)ceilf(log2f(amax / 6.0f));
    int ebest = 0; float errbest = -1.0f;
    for (int dd = -1; dd <= 4; dd++) {
        int ee = e0 + dd;
        if (ee < -126) ee = -126;            /* e8m0 字节 = e+127, 0 号是次正规特例, 255 是 NaN 槽 */
        if (ee > 127) ee = 127;
        const float sc = ldexpf(1.0f, ee);
        const float r = ds4_e2m1fn_round(v[j] / sc) * sc - v[j];
        sh[j] = r * r;
        __syncthreads();
        for (int o = 16; o > 0; o >>= 1) { if (j < o) sh[j] += sh[j + o]; __syncthreads(); }
        const float err = sh[0];             /* 所有线程读同一个归约结果 ⇒ ebest 全块一致 */
        if (errbest < 0.0f || err < errbest) { errbest = err; ebest = ee; }
        __syncthreads();
    }
    const uint8_t sb = (uint8_t)(ebest + 127);
    const float sc = ds4_e8m0_to_f32(sb);    /* ★解回值走 e8m0 解码★: 与主机解码器逐位同 */
    nb[j] = ds4_fp4_f32_to_nibble(v[j] / sc);
    __syncthreads();
    if (j < 16) blk[j] = (uint8_t)(nb[2 * j] | (uint8_t)(nb[2 * j + 1] << 4));
    if (j == 0) blk[16] = sb;
    s[base + j] = 1.0f + ds4_fp4_nibble_to_f32(nb[j]) * sc;
}

/* 第③步: 按【格点值】与解前值的差更新残差 —— 后面的专家从此看见的是含本专家量化误差的残差。 */
__global__ static void gr_resid_fix_kernel(float *resid, const float *s, const float *s_old,
                                           const float *ye, const float *rw,
                                           const int *rows, int nrow, int e, int nu, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float ds = s[(size_t)e * D + d] - s_old[d];
    if (ds == 0.f) return;
    for (int j = 0; j < nrow; j++) {
        const int pk = rows[j], i = pk / nu;
        resid[(size_t)i * D + d] -= ds * rw[(size_t)pk] * ye[(size_t)pk * D + d];
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
 * ds[n_expert][D] 输出缩放因子(锚 1), ★全部住在 fp4 格点上★; dpk[n_expert × D/32 × 17] 输出与之
 * 严格对应的落盘码字(调用方直接写文件, 不再做任何转换)。
 * out_train/out_val = 1 − ‖残差‖/‖靶‖ 的百分比(与现役放大器同口径, 幅值比不是平方能量比),
 * 因为最终残差是拿格点值算的, 这两个数就是【落地态】的读数。返回 0 成功。 */
extern "C" int v41_gr_solve_layer_gpu(const float *dye, const float *drw, const int *dsel,
                                      const float *dyfp, const float *dysh,
                                      int n, int nfit, int nu, int D, int n_expert,
                                      float lam, int niter, float *ds, uint8_t *dpk,
                                      float *out_train, float *out_val) {
    const size_t nD = (size_t)n * D, npair = (size_t)n * nu;
    if (D % 32) { fprintf(stderr, "★D=%d 不是 32 的整数倍, fp4x32 块装不下★\n", D); return -1; }
    float *resid = NULL, *dbar = NULL, *sold = NULL; double *dsum = NULL;
    GRCK(cudaMalloc((void **)&resid, nD * 4));
    GRCK(cudaMalloc((void **)&dbar, (size_t)D * 4));
    GRCK(cudaMalloc((void **)&sold, (size_t)D * 4));
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

    /* ② Gauss-Seidel 轮转, ★每个专家解完立刻落到 fp4 格点★(三发: 解 → 投影+出码字 → 按格点值修残差)。
     * 没有数据支撑的专家(nrow<=0)一次都不解, s 保持 1 —— 1 在"存 s−1"的口径下是精确的 0, 格点上有,
     * 所以这些专家的码字也是对的(全零块), 不需要额外处理。 */
    for (int it = 0; it < niter; it++) {
        for (int e = 0; e < n_expert; e++) {
            const int nrow = off[e + 1] - off[e];
            if (nrow <= 0) continue;
            gr_expert_kernel<<<(D + 255) / 256, 256>>>(ds, sold, resid, dye, drw, drows + off[e], nrow, e, lam, nu, D, dbar);
            gr_fp4_quant_kernel<<<D / 32, 32>>>(ds, dpk, e, D);
            gr_resid_fix_kernel<<<(D + 255) / 256, 256>>>(resid, ds, sold, dye, drw, drows + off[e], nrow, e, nu, D);
        }
        GRCK(cudaGetLastError());
        GRCK(cudaDeviceSynchronize());
    }
    /* 没被解过的专家也要有码字(s≡1 ⇒ 全零块), 否则 dpk 里是上一层留下的字节 */
    for (int e = 0; e < n_expert; e++)
        if (off[e + 1] - off[e] <= 0) gr_fp4_quant_kernel<<<D / 32, 32>>>(ds, dpk, e, D);
    GRCK(cudaGetLastError());
    GRCK(cudaDeviceSynchronize());

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
    cudaFree(drows); cudaFree(resid); cudaFree(dbar); cudaFree(sold); cudaFree(dsum);
    return 0;
}

#include "v41_gr_margin.inc.cu"   /* 后训练(第三件)的决策差解算: mg_* 一族 */
