/* v41_amp_diag.inc.cu — 放大器的逐行/逐通道诊断 + 输出通道白化核(2026-09-12)。单 TU 分片: 由
 * v41_amp_solve.cu #include, 共用它的 sub_kernel / amp_apply_core / g_hb(全是 static)。拆出来只因主文件超 500 行仓规。 */

/* ---------------- 逐行诊断: 能量被谁占了 ----------------
 * 【为什么要这个】2026-09-12 金融反修: 40 层逐层 held-out 能量增益全正(平均 +9.1%), 端到端
 * 却从 PPL 比 1.453 崩到 2.380。能量口径 ‖Y0−Δ‖²/‖Y0‖² 是按行平方和加权的 —— 几个巨值行
 * (massive 激活 token)能占掉大半能量, 最小二乘就去伺候它们, 普通行反被 20% 幅值的修正当噪声
 * 打(memory massive_channel_mine: 深层判读必须归一化口径)。本函数把每行的 ‖靶‖/‖残差‖/‖y_q‖
 * 都回给调用方, 让"能量增益"和"逐行增益"并排看 —— 分叉本身就是定罪证据。 */
__global__ static void rownorm_kernel(const float *Y, int N, int D, float *out) {
    __shared__ float sh[256];
    int r = blockIdx.x;
    if (r >= N) return;
    const float *y = Y + (long long)r * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) s += y[j] * y[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[r] = sqrtf(sh[0]);
}

/* 列(通道)范数: out[j] = ‖Y[:, j]‖, 一线程一列, 行主序 [N,D] 按列步进读 */
__global__ static void colnorm_kernel(const float *Y, int N, int D, float *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float s = 0.f;
    for (int i = 0; i < N; i++) { float v = Y[(long long)i * D + j]; s += v * v; }
    out[j] = sqrtf(s);
}

/* row_*: 每行范数(N 个)  col_*: 每通道范数(D 个), 同一套 靶/残差/y_q。
 * 行账查"能量是否被少数 token 垄断", 列账查"能量是否被少数通道(massive channel)垄断" ——
 * 后者决定 SVD 截断把秩花在了哪些方向上(列加权不改变满秩最小二乘解, 只改变截断保留谁)。 */
extern "C" int v41_amp_rowdiag_gpu(const void *dXv, const void *dYfpv, const void *dYqv,
                                   const void *dAv, const void *dBv, int N, int D, int K,
                                   float *row_tgt, float *row_res, float *row_yq,
                                   float *col_tgt, float *col_res, float *col_yq) {
    if (!g_hb) { if (cublasCreate(&g_hb) != CUBLAS_STATUS_SUCCESS) return -1; }
    float *dY0, *dZ, *dAB, *dR, *dC;
    CK(cudaMalloc(&dY0, sizeof(float) * (size_t)N * D));
    CK(cudaMalloc(&dZ,  sizeof(float) * (size_t)N * D));
    CK(cudaMalloc(&dAB, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dR,  sizeof(float) * N));
    CK(cudaMalloc(&dC,  sizeof(float) * D));
    const unsigned cb = (unsigned)((D + 255) / 256);
    rownorm_kernel<<<N, 256>>>((const float *)dYqv, N, D, dR);
    CK(cudaMemcpy(row_yq, dR, sizeof(float) * N, cudaMemcpyDeviceToHost));
    colnorm_kernel<<<cb, 256>>>((const float *)dYqv, N, D, dC);
    CK(cudaMemcpy(col_yq, dC, sizeof(float) * D, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dY0, dYfpv, sizeof(float) * (size_t)N * D, cudaMemcpyDeviceToDevice));
    sub_kernel<<<(unsigned)(((size_t)N * D + 255) / 256), 256>>>(dY0, (const float *)dYqv, (long long)N * D);
    rownorm_kernel<<<N, 256>>>(dY0, N, D, dR);
    CK(cudaMemcpy(row_tgt, dR, sizeof(float) * N, cudaMemcpyDeviceToHost));
    colnorm_kernel<<<cb, 256>>>(dY0, N, D, dC);
    CK(cudaMemcpy(col_tgt, dC, sizeof(float) * D, cudaMemcpyDeviceToHost));
    amp_apply_core((const float *)dXv, (const float *)dAv, (const float *)dBv, N, D, K, dAB, dZ);
    sub_kernel<<<(unsigned)(((size_t)N * D + 255) / 256), 256>>>(dY0, dZ, (long long)N * D);
    rownorm_kernel<<<N, 256>>>(dY0, N, D, dR);
    CK(cudaMemcpy(row_res, dR, sizeof(float) * N, cudaMemcpyDeviceToHost));
    colnorm_kernel<<<cb, 256>>>(dY0, N, D, dC);
    CK(cudaMemcpy(col_res, dC, sizeof(float) * D, cudaMemcpyDeviceToHost));
    cudaFree(dY0); cudaFree(dZ); cudaFree(dAB); cudaFree(dR); cudaFree(dC);
    return 0;
}

/* ---------------- 输出通道白化(2026-09-12 金融反修分叉的修法) ----------------
 * 【机理】满秩最小二乘解 W 对输出列是逐列独立的, 给输出通道加权不改变 W; 但 SVD 截断到 K 保留的
 * 是 W 里奇异值最大的方向 —— 原始空间里那就是几个巨值通道(massive channel)的方向。秩全花在
 * 巨值通道上, 普通通道的误差原样留下; 而 RMSNorm 之后巨值通道的单位误差对下游最不值钱。
 * 结果 = 层内能量账大涨(能量就是巨值通道的), 端到端反而退。
 * 【做法】W̃ = W·diag(1/σ) 后再 SVD 截断, 应用时 A 的列乘回 σ:  Δ = X·U_K S_K V_Kᵀ·diag(σ)。
 * σ_b = 拟合段上靶(whiten=1)或 y_fp(whiten=2)第 b 通道的 RMS; whiten=0 = 原始行为。
 * 白化只改截断保留谁, 不改满秩解; 它是"最优的那 K 个方向"的口径从能量换成了归一化误差。 */
__global__ static void scale_cols_kernel(float *W, int D, const float *s) {     /* 列主序 D×D, 列 b 乘 s[b] */
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long long)D * D) W[i] *= s[i / D];
}
__global__ static void scale_chan_kernel(float *Y, long long n, int D, const float *s) { /* 行主序 [N,D], 通道 j 乘 s[j] */
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] *= s[i % D];
}
__global__ static void rms_from_colnorm_kernel(float *s, int D, float inv_sqrt_n, int invert) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float v = s[j] * inv_sqrt_n;
    if (v < 1e-6f) v = 1e-6f;                  /* 全零通道: 不放大, 当它不存在 */
    s[j] = invert ? 1.f / v : v;
}
/* 按 whiten 模式算 σ(D 个, 设备内存)与 1/σ。src 行主序 [NFIT, D] */
static void channel_sigma(int whiten, const float *dY0f, const float *dYfpf, int NFIT, int D, float *dSig, float *dInv) {
    const unsigned cb = (unsigned)((D + 255) / 256);
    colnorm_kernel<<<cb, 256>>>(whiten == 2 ? dYfpf : dY0f, NFIT, D, dSig);
    CK(cudaMemcpy(dInv, dSig, sizeof(float) * D, cudaMemcpyDeviceToDevice));
    rms_from_colnorm_kernel<<<cb, 256>>>(dSig, D, 1.f / sqrtf((float)NFIT), 0);
    rms_from_colnorm_kernel<<<cb, 256>>>(dInv, D, 1.f / sqrtf((float)NFIT), 1);
}


/* ---------------- 第七版低秩形态(2026-10-01)要的两个行级小核 ----------------
 * rowdot:   out[i] = A[i]·B[i](一行一个 block, 共享内存归约, 与 rownorm_kernel 同构)。用途: ‖c_i‖² 与预测 Δlog p_i = c_i·Δ_i。
 * rowscale: dst[i][:] = src[i][:] × s[i] —— 把方向表 c_i 缩成靶向量 R_i = η·A_i·c_i/‖c_i‖²(于是 c_i·R_i = η·A_i 严格成立)。
 * 为什么不在主机做: 行数到 1.2 万、D=5120 时是 6e7 个元素, 来回搬一次 0.25 GB; 且本仓铁律新代码只走 GPU。 */
__global__ static void rowdot_kernel(const float *A, const float *B, int N, int D, float *out) {
    __shared__ float sh[256];
    int r = blockIdx.x;
    if (r >= N) return;
    const float *a = A + (long long)r * D, *b = B + (long long)r * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) s += a[j] * b[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[r] = sh[0];
}
__global__ static void rowscale_kernel(float *dst, const float *src, const float *s, int N, int D) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)N * D) return;
    dst[t] = src[t] * s[t / D];
}
extern "C" int v41_amp_rowdot_gpu(const void *dAv, const void *dBv, int N, int D, float *out_host) {
    float *dO;
    CK(cudaMalloc(&dO, sizeof(float) * (size_t)N));
    rowdot_kernel<<<N, 256>>>((const float *)dAv, (const float *)dBv, N, D, dO);
    CK(cudaMemcpy(out_host, dO, sizeof(float) * (size_t)N, cudaMemcpyDeviceToHost));
    cudaFree(dO);
    return 0;
}
extern "C" int v41_amp_rowscale_gpu(void *dDst, const void *dSrc, const float *scale_host, int N, int D) {
    float *dS;
    CK(cudaMalloc(&dS, sizeof(float) * (size_t)N));
    CK(cudaMemcpy(dS, scale_host, sizeof(float) * (size_t)N, cudaMemcpyHostToDevice));
    rowscale_kernel<<<(unsigned)(((size_t)N * D + 255) / 256), 256>>>((float *)dDst, (const float *)dSrc, dS, N, D);
    CK(cudaDeviceSynchronize());
    cudaFree(dS);
    return 0;
}
