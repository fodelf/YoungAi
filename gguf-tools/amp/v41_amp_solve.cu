/* v41_amp_solve.cu — V4.1 反修: 逐层解低秩放大器(2026-09-11)。
 *
 * 【这是什么】量化让每层 MoE 的输出偏了, 反修就是给每层配一个小的修正项 z(x) 把它拉回来:
 *     y_fixed = y_q + B·(A·x)     A[K,dim]  B[dim,K]   K=64
 * 这是 V4 时代 zlayer 的同一件事, 在 V4.1 的结构上重新实现 —— 原来那套硬编码了
 * DIM 4096 / MOEI 2048 / NEXP 256 / 43 层, 一行都跑不了 V4.1。
 *
 * 【★回归的输入必须是部署时的 x★ —— 2026-09-11 血教训】
 * 首版按"层内解算"的思路做: 靶与输入都取教师的 x(--force-x 层间隔断), 理由是"剔掉上游
 * 误差, 只修当层量化"。结果端到端 PPL 1.66 → 10311 崩得比不修还惨。
 * 根因 = ★x 错位税★(memory amp_inlayer_vs_e2e_verdict 早有记录, 40×): 放大器在"教师 x 分布"
 * 上拟合, 部署时每层喂进来的却是**量化态自己传播的 x_q**, 分布不同 ⇒ 修正偏; 40 层
 * 逐层累加偏量 ⇒ 完全失控。
 * 正修一: 回归输入改用 **x_q**(量化态自然传播, 不隔断)。这样解算时看到的 x 与部署时
 * 喂的 x 同分布; 靶里含的上游传播误差**正是端到端要修的东西**, 剔掉它才是错的。
 *
 * 【★靶的两端必须喂同一个 x★ —— 2026-09-11 第二刀, 上面那次只修了一半】
 * 输入改对之后端到端仍崩(判决语料 wt2 PPL 比值 55837, Σmin 0.0175)。真因: 靶写成
 *     y_fp(x_fp) − y_q(x_q)        ← 两端的 x 根本不是同一个
 * y_fp 是 FP 模型【自然跑】时 dump 的, 它的 x 是 FP 态; 而解算喂的 x 是量化态。
 * 这个差里混进了"x 从 x_fp 漂到 x_q 导致的输出变化", 那部分不是 x_q 的函数, 放大器
 * 无论多大的秩都拟合不掉 —— 实测每层只吃掉 13.6% 能量, 剩下 86% 就是它;
 * 而拟合料 PPL 比值 1.051 / 判决料 55837 的巨大分叉, 正是把这堆不可约成分当噪声
 * 背下来的过拟合签名。
 * 正修二: 靶改成 **y_fp(x_q) − y_q(x_q)** —— 同一个 x, 只有专家权重不同。
 * 序贯时 x 逐层变(前面层修正后传播), 没法预先 dump ⇒ 必须在同一次前向里当场
 * 用 FP 专家把 y_fp(x) 算出来(见 v41_amp_hooks.py: 关量化开关重跑一遍 ffn)。
 * ⇒ 离线 main(): x 与 y_q 从 <q目录> 读, y_fp 从 <fp目录> 读 —— 该目录必须是
 *   **--force-x 喂了 q 态 x 的 FP 前向**产物, 不能是 FP 自然跑的那份。
 *
 * 【★闭式, 不是训练★】min‖Y − W·X‖² 的最小二乘解 W = Y·Xᵀ(X·Xᵀ + λI)⁻¹, 再对 W 做
 * SVD 截断到秩 K, 得 B = U_K·S_K, A = V_Kᵀ。这是 reduced-rank regression 的标准闭式解,
 * 无梯度、无迭代、确定性。λ 是岭阻尼(X·Xᵀ 的迹的 1e-3), 防病态。
 *
 * 【体积】每层 (K·dim + dim·K)·f16 = 2·5120·64·2 = 1.31 MB, 40 层 52 MB ——
 * 对 116 GB 的模型是 0.045%, bpw 影响可忽略。
 *
 * 编译: nvcc -O3 -fmad=false -o v41_amp_solve v41_amp_solve.cu -lcublas -lcusolver
 * 用法: v41_amp_solve <fp目录> <q目录> <出目录> <层数> <dim> <ntok> [K=64]
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"★CUDA %s @%d: %s★\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"★cuBLAS %s @%d: %d★\n",#x,__LINE__,(int)s_); exit(1);} } while(0)
#define CS(x) do { cusolverStatus_t s_=(x); if(s_!=CUSOLVER_STATUS_SUCCESS){ \
    fprintf(stderr,"★cuSOLVER %s @%d: %d★\n",#x,__LINE__,(int)s_); exit(1);} } while(0)

static float *load_bin(const char *path, long long n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "★打不开 %s★\n", path); exit(1); }
    float *p = (float *)malloc(sizeof(float) * n);
    if (fread(p, 4, n, f) != (size_t)n) { fprintf(stderr, "★%s 读不满 %lld★\n", path, n); exit(1); }
    fclose(f);
    return p;
}

/* Y -= Yq (原地), 同时报残差能量 —— 靶的量级是判断反修有没有意义的第一眼 */
__global__ static void sub_kernel(float *Y, const float *Yq, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] -= Yq[i];
}

__global__ static void addlam_kernel(float *G, int d, float lam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) G[(long long)i * d + i] += lam;
}

/* B = U[:, :K] * S[:K]  (列主序 U 是 d×d) */
__global__ static void scale_u_kernel(const float *U, const float *S, float *B, int d, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)d * K) return;
    int col = (int)(i / d);
    B[i] = U[i] * S[col];
}

/* ---------------- 在线序贯接口 ----------------
 * 【为什么必须序贯】并行解算(所有层用同一次前向的状态)已被三次端到端实测证死:
 * PPL 265 → 10311 → 99972, 越修越崩。机理 = ★正反馈耦合★: L0 一修, x₁ 就变,
 * 而 L1 的放大器是按未修正的 x₁ 解的 ⇒ 失配; 失配的修正让 x₂ 偏更多 ⇒ 逐层放大。
 * 放大器的幅值还不小(吃掉 18% 能量 ⇒ z 约为靶的 42%, 而靶已达输出的 78~179%),
 * 等于每层往残差流注入 40~75% 输出幅值的扰动。
 * 序贯: 解 L_i 时 L0..L_{i-1} 已修正并传播 ⇒ 看到的 x 就是部署时的 x, 耦合自然闭合。
 * 本接口让调用方在【一次前向】里逐层解+立刻应用, 不必跑 40 趟。
 * handle 静态复用 —— 每层重建 cublas/cusolver handle 会吃掉大半时间。 */
static cublasHandle_t g_hb = NULL;
static cusolverDnHandle_t g_hs = NULL;

/* ★解算自检与运行时应用共用这一份★(2026-09-11): 同一个公式原来手写了三处
 * (这里的自检 + teacher 里 --amp/--amp-online 两处 Python 矩阵乘), 布局稍有出入
 * 就是一次静默错位 —— 方向写反那次端到端 PPL 从 1.66 崩到 265。
 *   Z[N,D] = X·(B·A)
 * 布局(cublas 全列主序, 但入参按内存布局描述):
 *   dX  行主序 [N,D] —— 列主序看是 D×N, 即 Xᵀ, 每列一个 token
 *   dB  列主序 D×K   —— 数学上的 B(D×K)
 *   dA  行主序 [K,D] —— 列主序看是 D×K, 转置后即 A(K×D)
 *   dZ  行主序 [N,D] —— 列主序算出 [D,N] = (X·B·A)ᵀ, 内存上正是 [N,D]
 * dAB 是调用方给的 D×D 暂存(省一次 malloc/free)。 */
static void amp_apply_core(const float *dX, const float *dA, const float *dB,
                           int N, int D, int K, float *dAB, float *dZ) {
    const float a1 = 1.f, b0 = 0.f;
    CB(cublasSgemm(g_hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, K, &a1, dB, D, dA, D, &b0, dAB, D));
    CB(cublasSgemm(g_hb, CUBLAS_OP_T, CUBLAS_OP_N, D, N, D, &a1, dAB, D, dX, D, &b0, dZ, D));
}

/* 运行时应用: Y += X·(B·A), 原地累加进 dYv[N,D](行主序)。
 * out_dz 非空时回填 ‖Δ‖(修正量的 F 范数) —— 信任域诊断量: 用它和 ‖Y‖ 比,
 * 就知道夹持该不该存在, 而不是拍脑袋加一个 clip 兜底。 */
extern "C" int v41_amp_apply_gpu(const void *dXv, void *dYv, const void *dAv, const void *dBv,
                                 int N, int D, int K, float *out_dz) {
    if (!g_hb) { if (cublasCreate(&g_hb) != CUBLAS_STATUS_SUCCESS) return -1; }
    float *dAB, *dZ;
    CK(cudaMalloc(&dAB, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dZ, sizeof(float) * (size_t)N * D));
    amp_apply_core((const float *)dXv, (const float *)dAv, (const float *)dBv, N, D, K, dAB, dZ);
    if (out_dz) CB(cublasSnrm2(g_hb, N * D, dZ, 1, out_dz));
    const float a1 = 1.f;
    CB(cublasSaxpy(g_hb, N * D, &a1, dZ, 1, (float *)dYv, 1));
    cudaFree(dAB); cudaFree(dZ);
    return 0;
}

#include "v41_amp_diag.inc.cu"   /* 逐行/逐通道诊断(单 TU 分片) */

/* ---------------- K 扫描: 秩到底该取多少 ----------------
 * 【为什么要这个】K=64 是 V4 冠军配置直接搬来的, 在 V4.1 上从没论证过。而 rank-64
 * 实测只吃掉 5.4% 误差能量(全域 8192, 2026-09-11), 这时有两种可能, 应对方式完全相反:
 *   a) 奇异值谱衰减快 ⇒ K=64 已近天花板, 加大 K 没用, 问题在【方法】—— 量化误差
 *      根本不是 x 的低秩线性函数, 该换挂点/换形式, 继续调 K 是浪费;
 *   b) 谱很平 ⇒ 纯粹是秩不够, 加大 K 就能吃到更多。
 * 【为什么不打奇异值谱而是直接扫 K】谱的能量占比是【W 自己的】能量分布, 判据却是
 * 【残差降了多少】—— 两者之间隔着 X 的相关结构, 不等价。直接对每个 K 算一次
 * ‖Y − X·W_K‖/‖Y‖, 要什么就量什么。每个 K 两次 GEMM, 相对一次 SVD 可忽略。
 * 【体积/速度账, 给取舍用】每层 2·D·K 个 f16:
 *   K=64 →1.31 MB/层·52 MB 全模型·+0.0008 bpw·0.57% 解码带宽
 *   K=256→5.24 MB/层·210 MB·+0.0031 bpw·2.3%
 *   K=1024→21 MB/层·839 MB·+0.0123 bpw·9.1%(引擎 z^L 秩上限就是 1024)
 * 体积从来不是约束, 带宽才是 —— 放大器每 token 每层都要读一遍。
 *
 * 【★必须 held-out★ 2026-09-11 血教训】首版只在【拟合集自己】上量残差, 据此选了 K=1024,
 * 端到端 Σmin 从 0.6711 崩到 0.2431(PPL 比值 1.427 → 45.8)。原因: 大 K 在拟合集上永远
 * 吃得更多 —— 那正是过拟合的定义, 拿它选 K 等于用"谁更会背答案"挑学生。
 * 自由度账(早该算): 观测 N·D=42M 个标量, K=1024 的 A+B 是 10.5M 参数 = 观测的 25%,
 * 记住样本绰绰有余; K=64 只占 1.6%。N=8192 让 XᵀX 满秩(解决欠定), 但没解决过参数化。
 * ⇒ 本函数把 token 切两段: 前 NFIT 个解, 后 N−NFIT 个【只评估不参与解算】。
 *   两段来自同一次前向的同一个序贯态, 语义一致, 不需要第二次前向。
 *   NFIT 必须 > D, 否则拟合那半又欠定, 量出来的还是假账。
 *
 * Ks 按升序给, nk 个; out_train[i]/out_val[i] 回填该 K 在两段上的 残差/靶
 * (越小越好, 1.0=白干)。**只看 out_val** —— out_train 仅用来确认过拟合的幅度。 */
extern "C" int v41_amp_scan_k_gpu(const void *dXv, const void *dYfpv, const void *dYqv,
                                  int N, int D, int NFIT, const int *Ks, int nk,
                                  const float *lams, int nl,
                                  float *out_train, float *out_val, float *out_tgt_rel,
                                  int whiten, float *out_train_w, float *out_val_w) {
    if (!g_hb) { if (cublasCreate(&g_hb) != CUBLAS_STATUS_SUCCESS) return -1; }
    if (!g_hs) { if (cusolverDnCreate(&g_hs) != CUSOLVER_STATUS_SUCCESS) return -1; }
    const float *dX = (const float *)dXv;
    const float a1 = 1.f, b0 = 0.f, bn1 = -1.f;

    float *dY0, *dY, *dG, *dXY, *dU, *dVt, *dS, *dwork, *dB, *dW, *dZ, *dG0, *dXY0, *dSig, *dInv;
    int *dinfo;
    CK(cudaMalloc(&dSig, sizeof(float) * D));
    CK(cudaMalloc(&dInv, sizeof(float) * D));
    CK(cudaMalloc(&dY0, sizeof(float) * (size_t)N * D));
    CK(cudaMalloc(&dY,  sizeof(float) * (size_t)N * D));
    CK(cudaMalloc(&dZ,  sizeof(float) * (size_t)N * D));
    CK(cudaMalloc(&dG,  sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dXY, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dU,  sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dVt, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dW,  sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dB,  sizeof(float) * (size_t)D * D));   /* 最大 K=D, 按满的开 */
    CK(cudaMalloc(&dS,  sizeof(float) * D));
    CK(cudaMalloc(&dG0,  sizeof(float) * (size_t)D * D));   /* XᵀX 原件, λ 循环每轮复制 */
    CK(cudaMalloc(&dXY0, sizeof(float) * (size_t)D * D));   /* XᵀY 原件, 同上 */
    CK(cudaMalloc(&dinfo, sizeof(int)));

    /* 靶 Y0 = y_fp − y_q, 以及它相对输出幅值的大小(量化到底伤多重) */
    CK(cudaMemcpy(dY0, dYfpv, sizeof(float) * (size_t)N * D, cudaMemcpyDeviceToDevice));
    sub_kernel<<<(unsigned)(((size_t)N * D + 255) / 256), 256>>>(dY0, (const float *)dYqv, (long long)N * D);
    float tgt = 0, yq = 0;
    CB(cublasSnrm2(g_hb, N * D, dY0, 1, &tgt));
    CB(cublasSnrm2(g_hb, N * D, (const float *)dYqv, 1, &yq));
    if (out_tgt_rel) *out_tgt_rel = yq > 0 ? tgt / yq : 0.f;

    /* 切两段。X 行主序 [N,D] = 列主序 D×N(每列一个 token) ⇒ 切 token 就是切列,
     * 前 NFIT 列天然连续, 后一段只要把指针挪 NFIT·D 个 float。 */
    const int NVAL = N - NFIT;
    const float *dXf = dX,            *dXv2 = dX  + (size_t)NFIT * D;
    const float *dY0f = dY0;          const float *dY0v = dY0 + (size_t)NFIT * D;
    float tgt_f = 0, tgt_v = 0, tgt_fw = 1, tgt_vw = 1;
    CB(cublasSnrm2(g_hb, NFIT * D, dY0f, 1, &tgt_f));
    CB(cublasSnrm2(g_hb, NVAL * D, dY0v, 1, &tgt_v));
    const unsigned ndb = (unsigned)(((size_t)N * D + 255) / 256);
    if (whiten) {           /* 白化空间里的靶范数: 每通道乘 1/σ 后算(借 dY 作暂存) */
        channel_sigma(whiten, dY0f, (const float *)dYfpv, NFIT, D, dSig, dInv);
        CK(cudaMemcpy(dY, dY0, sizeof(float) * (size_t)N * D, cudaMemcpyDeviceToDevice));
        scale_chan_kernel<<<ndb, 256>>>(dY, (long long)N * D, D, dInv);
        CB(cublasSnrm2(g_hb, NFIT * D, dY, 1, &tgt_fw));
        CB(cublasSnrm2(g_hb, NVAL * D, dY + (size_t)NFIT * D, 1, &tgt_vw));
    }

    /* ★正规方程与 λ 无关, 只算一次★ —— 逐 λ 重算是 V4 时代就定罪过的头号浪费
     * (ds4_z.h: "G=XᵀX 与 B=XᵀR 不依赖 λ")。★只用拟合段★。 */
    CB(cublasSgemm(g_hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, NFIT, &a1, dXf, D, dXf, D, &b0, dG0, D));
    CB(cublasSgemm(g_hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, NFIT, &a1, dXf, D, dY0f, D, &b0, dXY0, D));
    float trace = 0;
    CB(cublasSasum(g_hb, D, dG0, D + 1, &trace));
    int lw = 0, lw2 = 0;
    CS(cusolverDnSgesvd_bufferSize(g_hs, D, D, &lw));
    CS(cusolverDnSpotrf_bufferSize(g_hs, CUBLAS_FILL_MODE_LOWER, D, dG0, D, &lw2));
    if (lw2 > lw) lw = lw2;
    CK(cudaMalloc(&dwork, sizeof(float) * lw));

    /* ★λ 网格★(2026-09-12 修): 原来 λ 硬编码 trace/D·1e-3, 无候选无择优 —— 而 V4 的
     * zloss_solve.c 一直是 {3e-3, 3e-2, 3e-1} 三档配 ranks[] 二维择优。λ 是压过拟合的
     * 唯一旋钮, 焊死在候选范围下端 300× 的位置去扫 K, 必然量出"train 单调涨 / val
     * 单调跌", 那是正则缺席的形态, 不是"靶不可学"。 */
    for (int lj = 0; lj < nl; lj++) {
      CK(cudaMemcpy(dG,  dG0,  sizeof(float) * (size_t)D * D, cudaMemcpyDeviceToDevice));
      CK(cudaMemcpy(dXY, dXY0, sizeof(float) * (size_t)D * D, cudaMemcpyDeviceToDevice));
      addlam_kernel<<<(D + 255) / 256, 256>>>(dG, D, trace / D * lams[lj]);
      CS(cusolverDnSpotrf(g_hs, CUBLAS_FILL_MODE_LOWER, D, dG, D, dwork, lw, dinfo));
      CS(cusolverDnSpotrs(g_hs, CUBLAS_FILL_MODE_LOWER, D, D, dG, D, dXY, D, dinfo));
      if (whiten) scale_cols_kernel<<<(unsigned)(((size_t)D * D + 255) / 256), 256>>>(dXY, D, dInv);  /* W̃ = W·diag(1/σ) */
      CS(cusolverDnSgesvd(g_hs, 'A', 'A', D, D, dXY, D, dS, dU, D, dVt, D, dwork, lw, NULL, dinfo));

      for (int i = 0; i < nk; i++) {
        int K = Ks[i] > D ? D : Ks[i];
        /* W_K = (U[:,:K]·diag(S[:K])) · Vt[:K,:] —— Vt 列主序 D×D, 前 K 行用 ldb=D 直接取 */
        scale_u_kernel<<<(unsigned)(((size_t)D * K + 255) / 256), 256>>>(dU, dS, dB, D, K);
        CB(cublasSgemm(g_hb, CUBLAS_OP_N, CUBLAS_OP_N, D, D, K, &a1, dB, D, dVt, D, &b0, dW, D));
        if (whiten) scale_cols_kernel<<<(unsigned)(((size_t)D * D + 255) / 256), 256>>>(dW, D, dSig);  /* 乘回 σ, 回原始空间 */
        /* 两段各算一次残差。Z = X·W_K, 与 amp_apply_core 同式(dW 已是 B·A 的合体)。
         * 原始空间残差照报(与历史可比); whiten 时另报白化空间残差(择优要看这个) */
        const float *segX[2] = {dXf, dXv2}; const float *segY[2] = {dY0f, dY0v};
        const int segN[2] = {NFIT, NVAL}; const float segT[2] = {tgt_f, tgt_v}, segTw[2] = {tgt_fw, tgt_vw};
        for (int sg = 0; sg < 2; sg++) {
            float res = 0;
            CB(cublasSgemm(g_hb, CUBLAS_OP_T, CUBLAS_OP_N, D, segN[sg], D, &a1, dW, D, segX[sg], D, &b0, dZ, D));
            CK(cudaMemcpy(dY, segY[sg], sizeof(float) * (size_t)segN[sg] * D, cudaMemcpyDeviceToDevice));
            CB(cublasSaxpy(g_hb, segN[sg] * D, &bn1, dZ, 1, dY, 1));
            CB(cublasSnrm2(g_hb, segN[sg] * D, dY, 1, &res));
            (sg ? out_val : out_train)[lj * nk + i] = segT[sg] > 0 ? res / segT[sg] : 1.f;
            if (whiten && out_train_w && out_val_w) {
                scale_chan_kernel<<<(unsigned)(((size_t)segN[sg] * D + 255) / 256), 256>>>(dY, (long long)segN[sg] * D, D, dInv);
                CB(cublasSnrm2(g_hb, segN[sg] * D, dY, 1, &res));
                (sg ? out_val_w : out_train_w)[lj * nk + i] = segTw[sg] > 0 ? res / segTw[sg] : 1.f;
            }
        }
      }
    }

    cudaFree(dY0); cudaFree(dY); cudaFree(dZ); cudaFree(dG); cudaFree(dXY);
    cudaFree(dU); cudaFree(dVt); cudaFree(dW); cudaFree(dB); cudaFree(dS);
    cudaFree(dG0); cudaFree(dXY0); cudaFree(dwork); cudaFree(dinfo); cudaFree(dSig); cudaFree(dInv);
    return 0;
}

/* 序贯解算。★lam 是 ridge 强度★(2026-09-12 加): 原来硬编码 1e-3, 那是 V4 候选范围
 * 下端的 1/300, 等于近乎无正则 —— held-out 扫描实测 λ=1e-3 时 K=64 的泛化是 −66.5%,
 * 而 λ=10 是 +1.8%。λ 是压过拟合的唯一旋钮, 必须由调用方按扫描结果给。
 * 扫描峰值(3 层平均, held-out): λ=10~30, K=128 → +2.0%(带宽 1.14%)。 */
extern "C" int v41_amp_solve_layer_gpu(const void *dXv, const void *dYfpv, const void *dYqv,
                                       int N, int D, int K, float lam,
                                       void *dAv, void *dBv, float *out_ratio, int whiten) {
    if (!g_hb) { if (cublasCreate(&g_hb) != CUBLAS_STATUS_SUCCESS) return -1; }
    if (!g_hs) { if (cusolverDnCreate(&g_hs) != CUSOLVER_STATUS_SUCCESS) return -1; }
    const float *dX = (const float *)dXv;
    float *dA = (float *)dAv, *dB = (float *)dBv;
    const float a1 = 1.f, b0 = 0.f, bn1 = -1.f;

    float *dY, *dG, *dXY, *dU, *dVt, *dS, *dwork, *dAB, *dZ, *dSig, *dInv;
    int *dinfo;
    CK(cudaMalloc(&dSig, sizeof(float) * D));
    CK(cudaMalloc(&dInv, sizeof(float) * D));
    CK(cudaMalloc(&dY, sizeof(float) * (size_t)N * D));
    CK(cudaMalloc(&dG, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dXY, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dU, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dVt, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dS, sizeof(float) * D));
    CK(cudaMalloc(&dinfo, sizeof(int)));
    /* 靶 Y = y_fp − y_q */
    CK(cudaMemcpy(dY, dYfpv, sizeof(float) * (size_t)N * D, cudaMemcpyDeviceToDevice));
    sub_kernel<<<(unsigned)(((size_t)N * D + 255) / 256), 256>>>(dY, (const float *)dYqv, (long long)N * D);
    float tgt = 0;
    CB(cublasSnrm2(g_hb, N * D, dY, 1, &tgt));

    CB(cublasSgemm(g_hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, N, &a1, dX, D, dX, D, &b0, dG, D));
    float trace = 0;
    CB(cublasSasum(g_hb, D, dG, D + 1, &trace));
    addlam_kernel<<<(D + 255) / 256, 256>>>(dG, D, trace / D * lam);
    CB(cublasSgemm(g_hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, N, &a1, dX, D, dY, D, &b0, dXY, D));

    int lw = 0, lw2 = 0;
    CS(cusolverDnSgesvd_bufferSize(g_hs, D, D, &lw));
    CS(cusolverDnSpotrf_bufferSize(g_hs, CUBLAS_FILL_MODE_LOWER, D, dG, D, &lw2));
    if (lw2 > lw) lw = lw2;
    CK(cudaMalloc(&dwork, sizeof(float) * lw));
    CS(cusolverDnSpotrf(g_hs, CUBLAS_FILL_MODE_LOWER, D, dG, D, dwork, lw, dinfo));
    CS(cusolverDnSpotrs(g_hs, CUBLAS_FILL_MODE_LOWER, D, D, dG, D, dXY, D, dinfo));
    float *hSig = (float *)malloc(sizeof(float) * D);
    for (int j = 0; j < D; j++) hSig[j] = 1.f;
    if (whiten) {           /* σ 从本次拟合行算(dY 此时仍是靶 Y0); W̃ = W·diag(1/σ) 再截断 */
        channel_sigma(whiten, dY, (const float *)dYfpv, N, D, dSig, dInv);
        scale_cols_kernel<<<(unsigned)(((size_t)D * D + 255) / 256), 256>>>(dXY, D, dInv);
        CK(cudaMemcpy(hSig, dSig, sizeof(float) * D, cudaMemcpyDeviceToHost));
    }
    CS(cusolverDnSgesvd(g_hs, 'A', 'A', D, D, dXY, D, dS, dU, D, dVt, D, dwork, lw, NULL, dinfo));
    scale_u_kernel<<<(unsigned)(((size_t)D * K + 255) / 256), 256>>>(dU, dS, dB, D, K);
    /* A = Vᵀ 前 K 行(白化时列 c 乘回 σ_c): dVt 列主序 D×D, 目标 dA 行主序 [K,D] ⇒ 逐元素搬 */
    {
        float *hVt = (float *)malloc(sizeof(float) * (size_t)D * D);
        float *hA = (float *)malloc(sizeof(float) * (size_t)K * D);
        CK(cudaMemcpy(hVt, dVt, sizeof(float) * (size_t)D * D, cudaMemcpyDeviceToHost));
        for (int c = 0; c < D; c++)
            for (int r = 0; r < K; r++) hA[(size_t)r * D + c] = hVt[(size_t)c * D + r] * hSig[c];
        CK(cudaMemcpy(dA, hA, sizeof(float) * (size_t)K * D, cudaMemcpyHostToDevice));
        free(hVt); free(hA);
    }
    free(hSig);
    /* 自检: 残差走 amp_apply_core —— 与运行时应用【同一份代码】, 不再手抄第二遍 */
    CK(cudaMalloc(&dAB, sizeof(float) * (size_t)D * D));
    CK(cudaMalloc(&dZ, sizeof(float) * (size_t)N * D));
    amp_apply_core(dX, dA, dB, N, D, K, dAB, dZ);
    CB(cublasSaxpy(g_hb, N * D, &bn1, dZ, 1, dY, 1));
    float res = 0;
    CB(cublasSnrm2(g_hb, N * D, dY, 1, &res));
    if (out_ratio) *out_ratio = tgt > 0 ? res / tgt : 1.f;

    cudaFree(dY); cudaFree(dG); cudaFree(dXY); cudaFree(dU); cudaFree(dVt);
    cudaFree(dS); cudaFree(dwork); cudaFree(dinfo); cudaFree(dAB); cudaFree(dZ); cudaFree(dSig); cudaFree(dInv);
    return (res < tgt) ? 0 : 1;      /* 1 = 放大器没用(残差没降), 调用方应跳过这一层 */
}

#ifndef V41_AMP_NOMAIN   /* 当对象链进 v41_amp_run(引擎路反修驱动)时不要 main; .so(Python 路)与独立程序照旧 */
int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: v41_amp_solve <fp目录(只取y)> <q目录(取x与y)> <出目录> <层数> <dim> <ntok> [K=64]\n");
        return 2;
    }
    const char *fpd = argv[1], *qd = argv[2], *outd = argv[3];
    int NL = atoi(argv[4]), D = atoi(argv[5]);
    long long N = atoll(argv[6]);
    int K = argc > 7 ? atoi(argv[7]) : 64;

    cublasHandle_t hb; CB(cublasCreate(&hb));
    cusolverDnHandle_t hs; CS(cusolverDnCreate(&hs));
    const float a1 = 1.f, b0 = 0.f, bn1 = -1.f;

    float *dX, *dY, *dYq, *dG, *dXY, *dW, *dU, *dVt, *dS, *dB;
    CK(cudaMalloc(&dX, sizeof(float) * N * D));
    CK(cudaMalloc(&dY, sizeof(float) * N * D));
    CK(cudaMalloc(&dYq, sizeof(float) * N * D));
    CK(cudaMalloc(&dG, sizeof(float) * (long long)D * D));
    CK(cudaMalloc(&dXY, sizeof(float) * (long long)D * D));
    CK(cudaMalloc(&dW, sizeof(float) * (long long)D * D));
    CK(cudaMalloc(&dU, sizeof(float) * (long long)D * D));
    CK(cudaMalloc(&dVt, sizeof(float) * (long long)D * D));
    CK(cudaMalloc(&dS, sizeof(float) * D));
    CK(cudaMalloc(&dB, sizeof(float) * (long long)D * K));

    int lwork = 0;
    CS(cusolverDnSgesvd_bufferSize(hs, D, D, &lwork));
    float *dwork; int *dinfo;
    CK(cudaMalloc(&dwork, sizeof(float) * lwork));
    CK(cudaMalloc(&dinfo, sizeof(int)));

    float *hB = (float *)malloc(sizeof(float) * (long long)D * K);
    float *hA = (float *)malloc(sizeof(float) * (long long)K * D);

    for (int L = 0; L < NL; L++) {
        char p1[512], p2[512], p3[512];
        snprintf(p1, sizeof p1, "%s/x_L%02d.bin", qd, L);   /* ★x 取量化态的, 不是教师的★ */
        snprintf(p2, sizeof p2, "%s/y_L%02d.bin", fpd, L);
        snprintf(p3, sizeof p3, "%s/y_L%02d.bin", qd, L);
        float *hx = load_bin(p1, N * D), *hy = load_bin(p2, N * D), *hyq = load_bin(p3, N * D);
        CK(cudaMemcpy(dX, hx, sizeof(float) * N * D, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dY, hy, sizeof(float) * N * D, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dYq, hyq, sizeof(float) * N * D, cudaMemcpyHostToDevice));
        free(hx); free(hy); free(hyq);

        /* 靶 Y = y_fp − y_q */
        sub_kernel<<<(unsigned)((N * D + 255) / 256), 256>>>(dY, dYq, N * D);
        float tgt = 0, ynorm = 0;
        CB(cublasSnrm2(hb, N * D, dY, 1, &tgt));
        CB(cublasSnrm2(hb, N * D, dYq, 1, &ynorm));

        /* 行主序 X[N,D] 在列主序里是 Xᵀ[D,N]。
         * G = X·Xᵀ(D×D): 列主序下 = Xᵀ_cm · (Xᵀ_cm)ᵀ ⇒ OP_N × OP_T, lda=D */
        CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, (int)N, &a1, dX, D, dX, D, &b0, dG, D));
        float trace = 0;
        CB(cublasSasum(hb, D, dG, D + 1, &trace));      /* 对角线步长 D+1 */
        float lam = trace / D * 1e-3f;
        addlam_kernel<<<(D + 255) / 256, 256>>>(dG, D, lam);

        /* XY = X·Yᵀ (D×D) */
        CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, (int)N, &a1, dX, D, dY, D, &b0, dXY, D));

        /* 解 G·Wᵀ = XY ⇒ W = XYᵀ·G⁻¹。G 对称正定 ⇒ Cholesky */
        int lw2 = 0;
        CS(cusolverDnSpotrf_bufferSize(hs, CUBLAS_FILL_MODE_LOWER, D, dG, D, &lw2));
        if (lw2 > lwork) { cudaFree(dwork); CK(cudaMalloc(&dwork, sizeof(float) * lw2)); lwork = lw2; }
        CS(cusolverDnSpotrf(hs, CUBLAS_FILL_MODE_LOWER, D, dG, D, dwork, lwork, dinfo));
        CS(cusolverDnSpotrs(hs, CUBLAS_FILL_MODE_LOWER, D, D, dG, D, dXY, D, dinfo));
        CK(cudaMemcpy(dW, dXY, sizeof(float) * (long long)D * D, cudaMemcpyDeviceToDevice));

        /* SVD 截断: W = U·S·Vᵀ, 取前 K */
        CS(cusolverDnSgesvd(hs, 'A', 'A', D, D, dW, D, dS, dU, D, dVt, D, dwork, lwork, NULL, dinfo));
        scale_u_kernel<<<(unsigned)(((long long)D * K + 255) / 256), 256>>>(dU, dS, dB, D, K);

        CK(cudaMemcpy(hB, dB, sizeof(float) * (long long)D * K, cudaMemcpyDeviceToHost));
        /* A = Vᵀ 的前 K 行; Vt 列主序 D×D, 取前 K 行 ⇒ 逐列挑 */
        float *hVt = (float *)malloc(sizeof(float) * (long long)D * D);
        CK(cudaMemcpy(hVt, dVt, sizeof(float) * (long long)D * D, cudaMemcpyDeviceToHost));
        for (int c = 0; c < D; c++)
            for (int r = 0; r < K; r++) hA[(long long)r * D + c] = hVt[(long long)c * D + r];
        free(hVt);

        char po[512];
        snprintf(po, sizeof po, "%s/amp_L%02d.bin", outd, L);
        FILE *f = fopen(po, "wb");
        if (!f) { fprintf(stderr, "★写不了 %s★\n", po); return 1; }
        int32_t hd[3] = { D, K, 1 };
        fwrite(hd, 4, 3, f);
        fwrite(hA, 4, (size_t)K * D, f);      /* A[K,D] 行主序 */
        fwrite(hB, 4, (size_t)D * K, f);      /* B[D,K] 列主序(= 行主序的 [K,D] 转置, 见应用方) */
        fclose(f);
        /* ★自检: 拟合后残差必须小于靶★
         * 这一步是 2026-09-11 的血教训: 应用侧把 Z = X·(B·A) 写成了 X·(B·A)ᵀ, 方向反了 ——
         * 矩阵方向错不会抛异常, 只会安静产出一个"能跑但有害"的放大器, 结果端到端 PPL 从 1.66
         * 崩到 265 才发现, 白烧一趟。解算器自己算一句残差, 当场就露馅(错向残差 > 靶 100%)。
         * 残差 = ‖Y − X·(B·A)‖, 用与应用侧【完全相同】的算式, 这样方向错了这里必然先炸。 */
        {
            float *dZ, *dAB;
            CK(cudaMalloc(&dZ, sizeof(float) * N * D));
            CK(cudaMalloc(&dAB, sizeof(float) * (long long)D * D));
            float *dA;
            CK(cudaMalloc(&dA, sizeof(float) * (long long)K * D));
            CK(cudaMemcpy(dA, hA, sizeof(float) * (long long)K * D, cudaMemcpyHostToDevice));
            /* AB = B·A (D×D): 列主序下 dB 是 [D,K], dA 行主序 [K,D] = 列主序 [D,K] */
            CB(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_T, D, D, K, &a1, dB, D, dA, D, &b0, dAB, D));
            /* Z = X·AB: 行主序 Z[N,D] ⇔ 列主序 Zᵀ[D,N] = ABᵀ · Xᵀ */
            CB(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, D, (int)N, D, &a1, dAB, D, dX, D, &b0, dZ, D));
            CB(cublasSaxpy(hb, (int)(N * D), &bn1, dZ, 1, dY, 1));   /* Y -= Z */
            float res = 0;
            CB(cublasSnrm2(hb, (int)(N * D), dY, 1, &res));
            cudaFree(dZ); cudaFree(dAB); cudaFree(dA);
            if (res >= tgt) {
                fprintf(stderr, "★L%02d 自检失败: 拟合后残差 %.4g ≥ 靶 %.4g —— 放大器没用或方向错, 硬停★\n",
                        L, res, tgt);
                return 3;
            }
            printf("L%02d 靶‖y_fp−y_q‖=%.4g (‖y_q‖=%.4g, 相对 %.1f%%), λ=%.3g, "
                   "拟合后残差 %.4g (吃掉 %.1f%% 能量) → %s\n",
                   L, tgt, ynorm, 100.0 * tgt / (ynorm > 0 ? ynorm : 1), lam,
                   res, 100.0 * (1.0 - (double)res / tgt), po);
        }
        if (0) printf("L%02d 靶=%.4g λ=%.3g → %s\n", L, tgt, lam, po);
        fflush(stdout);
    }
    printf("★%d 层放大器已解(K=%d, 每层 %.2f MB, 合计 %.1f MB)★\n",
           NL, K, 2.0 * D * K * 2 / 1e6, NL * 2.0 * D * K * 2 / 1e6);
    return 0;
}
#endif /* V41_AMP_NOMAIN */
