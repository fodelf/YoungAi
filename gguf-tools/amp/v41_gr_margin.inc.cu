/* v41_gr_margin.inc.cu — 后训练(三文件的第三件)的决策差解算, v41_gr_solve.cu 单 TU include。
 * 拆出来只为守单文件 ≤500 行; 与权重侧反修(gr_*)共用同一套取料缓冲与 GRCK 宏, 语义完全独立:
 * gr_* 解的是"整层输出像不像 FP 教师"(每行 D 条方程), mg_* 解的是"这一个决策点翻不翻"(每点 1 条)。 */
/* ======================= 后训练(第三件)的第二种靶: 决策差 =======================
 * 反修解的是"这一层的输出整体像不像 FP 教师"—— 每行 D=5120 个方程。后训练要的完全不是这个:
 * 一个决策点翻不翻, 只由 ★一个数★ 决定 ——
 *     m_i = ℓ_i[对版 token] − ℓ_i[错版 token]   (m>0 ⇒ 贪心写对版)
 * 末层之后就是 norm→head, 没有别的子层, 所以 m 对本层 MoE 输出是【精确线性】的:
 *     Δm_i = Σ_k Σ_d Δs[e_ik][d] · (rw_ik · ye_ik[d]) · c_i[d],
 *     c_i[d] = α_i · inv_i · γ[d] · (W[对][d] − W[错][d])     ← 调用方按 back.md §4.1 备好
 * (inv = 出口 RMSNorm 的 rsqrt 标量, 引擎 --score-rms 给; rms 随 Δ 变但它是逐 token 正标量,
 *  同乘所有 logit ⇒ 不改 m 的符号, 只是二阶的幅值误差。)
 *
 * ★为什么非要换这个靶★ 09-13 段 1 实撞: 拿"损失梯度"当靶在 y 空间解, train 只挽回 0.20%,
 * 缩放因子却已经跑到 [0.29, 2.00] —— 梯度是由目标 token 词向量定的稠密方向, 与"选了哪 6 个专家"
 * 无结构关系, 增益形态表达不了; 而那 5119 个无关方向占了靶能量的 99.8%。这里每个决策点只写 1 条方程。
 *
 * 【解什么】min_Δ Σ_决策 (a_i·Δ − b_i)² + ρ Σ_约束 (a_j·Δ)² + λ Σ_ed ridge[d]·Δ[e][d]²
 *   决策行 b_i = τ − m_i⁰(想把 logit 差推到 τ); 约束行 b=0(= "这一行原本最想说的两个 token 的差别动")。
 *   ridge[d] = λ · 本通道数据对角在专家上的平均 —— 与反修 den_bar 同精神(冷门专家被拉回不改),
 *   但单位跟着 c 走, 所以 λ 的量纲在两条路上是一致的。
 * 【怎么解】未知数 1.97M, 但矩阵一个字节都不建: 共轭梯度, 每次迭代两趟 —— A·x(一行一个 block 归约)
 *   与 Aᵀr(逐 (i,d) 散射 atomicAdd)。读的就是已经在设备上的 ye(0.93 GB)+c(0.16 GB)。 */

/* ★"对"而不是"行"★(2026-09-13 夜第三针): 一个方程 = 一个 (行, 要比的两个 token) 组合。
 * 同一行可以出多条方程 —— 修正作用在整个 5120 维通道上, 会把一堆 token 的 logit 一起抬,
 * 每行只钉一对的后果实测过: 钉住的那一对分毫不差(自检 2 相关 1.0000), 而 37% 的位置
 * argmax 还是变了, 因为冒头的是第三个 token。src[p] = 这一对取哪一行的料。
 *
 * m[p] = Σ_k Σ_d x[e][d]·rw·ye·C_p[d], 料取第 src[p] 行。一个对一个 block。 */
/* ★x 键控(2026-09-14, back.md 段 4)★: 未知数从 [e][d] 扩成 [r][e][d], 第 r 份的作用被
 * 门控标量 gate[i][r] = σ(q_r·x_i) 调制(r=0 是恒等门 gate≡1 ⇒ ★R=1 严格退化成旧形态★)。
 * 为什么非要它: 09-14 三针诊断判了旧形态的死刑 —— 决策点的靶方向 c_i 两两 cos +0.015、
 * 同号 51.8%(纯随机), 专家重合却高达 94.6%。也就是说"选了哪些专家"这个唯一的条件槽在这批
 * 数据上区分度≈0, 而每个教训要往哪个词推又互不相干 ⇒ 常数增益只能表达"无条件偏爱某几个词",
 * 折外实测撬动 0 个、弄坏 2~6 个。门控就是给形态补一个"在什么情况下"的槽。
 * ★给定 q_r 之后 σ(q_r·x_i) 是每行一个已知标量, 整个问题仍是线性最小二乘★ —— 所以 CG、
 * 主动集、尺 L 一个字不用改, 不动引擎就能先判它有没有戏。 */
__global__ static void mg_ax_kernel(float *m, const float *x, const float *ye, const float *rw,
                                    const int *sel, const float *C, const int *src, int nu, int D, int est,
                                    const float *gate, int R, size_t xsz) {
    __shared__ double sh[256];
    const int p = blockIdx.x, i = src[p];
    const float *ci = C + (size_t)p * D;
    double acc = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        double t = 0.0;
        for (int k = 0; k < nu; k++) {
            const int e = sel[(size_t)i * nu + k];
            if (e < 0) continue;
            const double u = (double)rw[(size_t)i * nu + k] * ye[((size_t)i * nu + k) * (size_t)D + d];
            double sr = 0.0;
            for (int r = 0; r < R; r++)
                sr += (double)gate[(size_t)i * R + r] * x[(size_t)r * xsz + (size_t)e * est + d];
            t += u * sr;
        }
        acc += t * ci[d];
    }

    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) m[p] = (float)sh[0];
}

/* out[e][d] += Σ_p v_p · (rw·ye·C_p[d])。v = 加权残差; sq!=0 时累加平方(建数据对角用)。 */
__global__ static void mg_atr_kernel(float *out, const float *v, const float *ye, const float *rw,
                                     const int *sel, const float *C, const int *src, int npair, int nu, int D, int sq, int est,
                                     const float *gate, int R, size_t xsz) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, p = blockIdx.y;
    if (d >= D) return;
    (void)npair;
    const int i = src[p];
    const float c = C[(size_t)p * D + d], vp = v[p];
    if (c == 0.f || vp == 0.f) return;
    for (int k = 0; k < nu; k++) {
        const int e = sel[(size_t)i * nu + k];
        if (e < 0) continue;
        const float a0 = rw[(size_t)i * nu + k] * ye[((size_t)i * nu + k) * (size_t)D + d] * c;
        for (int r = 0; r < R; r++) {
            const float a = a0 * gate[(size_t)i * R + r];
            atomicAdd(&out[(size_t)r * xsz + (size_t)e * est + d], sq ? a * a * vp : a * vp);
        }
    }
}

/* Δy[i][d] = Σ_k Δs[e_ik][d]·rw·ye —— 本层 MoE 输出被这一版修正挪了多少。主动集扫榜要它
 * (拿它去乘出口头, 看榜上每个 token 的 logit 各动了多少)。 */
extern "C" int v41_gr_margin_dy_gpu(const float *dye, const float *drw, const int *dsel, const float *dx,
                                    int n, int nu, int D, int n_expert, const float *dgate, int R, float *dY);
__global__ static void mg_dy_kernel(float *Y, const float *x, const float *ye, const float *rw,
                                    const int *sel, int nu, int D, int est, const float *gate, int R, size_t xsz) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < nu; k++) {
        const int e = sel[(size_t)i * nu + k];
        if (e < 0) continue;
        float sr = 0.f;
        for (int r = 0; r < R; r++) sr += gate[(size_t)i * R + r] * x[(size_t)r * xsz + (size_t)e * est + d];
        acc += sr * rw[(size_t)i * nu + k] * ye[((size_t)i * nu + k) * (size_t)D + d];
    }
    Y[(size_t)i * D + d] = acc;
}
extern "C" int v41_gr_margin_dy_gpu(const float *dye, const float *drw, const int *dsel, const float *dx,
                                    int n, int nu, int D, int n_expert, const float *dgate, int R, float *dY) {
    mg_dy_kernel<<<dim3((D + 255) / 256, n), 256>>>(dY, dx, dye, drw, dsel, nu, D, n_expert > 1 ? D : 0,
                                                    dgate, R, (size_t)n_expert * D);
    GRCK(cudaGetLastError());
    GRCK(cudaDeviceSynchronize());
    return 0;
}

/* ridge[d] = lam · (Σ_e dd[e][d]) / n_expert; 同时把 precond M = dd + ridge·we 写回 dd。
 * ★we[e] = 专家频率岭★(2026-09-14, back.md §4.7a): NULL = 全 1(旧行为)。
 * 为什么要它: 岭只按通道能量加权时, 对"只在一个决策点出现过一次的冷门专家"和"几十行都在用的
 * 高频专家"要价一样。最小范数解于是乐意把修正全堆在冷门专家上 —— 那一行百分之百翻得动, 而新的
 * 一天几乎不会再选到它 ⇒ 纯记忆。we[e] = (f_max+κ)/(f_e+κ) 让冷门专家变贵, 解被推向反复出现的
 * 高频专家, 那些专家判决日也会被选中, 修正才谈得上迁移。 */
__global__ static void mg_ridge_kernel(float *dd, float *ridge, const float *we, int n_expert, int D, float lam, int R) {
    __shared__ double sh[256];
    const int d = blockIdx.x;
    double acc = 0.0;
    for (int t = threadIdx.x; t < n_expert * R; t += blockDim.x) acc += dd[(size_t)t * D + d];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    const float rg = (float)(lam * sh[0] / ((double)n_expert * R));
    if (threadIdx.x == 0) ridge[d] = rg;
    __syncthreads();
    /* M 的下限: 全没数据的 (e,d) 上 dd=0, 若 ridge 也是 0(整通道无能量)就会除零 */
    for (int t = threadIdx.x; t < n_expert * R; t += blockDim.x) {
        const int e = t % n_expert;
        const float m = dd[(size_t)t * D + d] + rg * (we ? we[e] : 1.f);
        dd[(size_t)t * D + d] = m > 0.f ? m : 1.f;
    }
}

__global__ static void mg_axpy(float *y, const float *x, float a, size_t n) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) y[t] += a * x[t];
}
__global__ static void mg_pz(float *p, const float *z, float b, size_t n) {   /* p = z + b·p */
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) p[t] = z[t] + b * p[t];
}
__global__ static void mg_precond(float *z, const float *r, const float *M, size_t n) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) z[t] = r[t] / M[t];
}
__global__ static void mg_ridge_add(float *hp, const float *p, const float *ridge, const float *we, int n_expert, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, e = t % n_expert;
    if (d < D) hp[(size_t)t * D + d] += ridge[d] * (we ? we[e] : 1.f) * p[(size_t)t * D + d];
}
__global__ static void mg_dot_kernel(double *out, const float *a, const float *b, size_t n) {
    __shared__ double sh[256];
    double acc = 0.0;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)blockDim.x * gridDim.x)
        acc += (double)a[t] * b[t];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}
/* rr[i] = w[i]·v[i] —— 行权重乘进去(W 是对角阵, 就这一句)。 */
__global__ static void mg_rowscale(float *rr, const float *v, const float *w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rr[i] = w[i] * v[i];
}

static double mg_dot(double *dsum, const float *a, const float *b, size_t n) {
    double h = 0.0;
    cudaMemset(dsum, 0, sizeof(double));
    mg_dot_kernel<<<256, 256>>>(dsum, a, b, n);
    cudaMemcpy(&h, dsum, sizeof(double), cudaMemcpyDeviceToHost);
    return h;
}

/* 只算 Δm(不解算): 落盘量化【之后】必须用它重算一遍预测 —— 产物是 fp4x32(4 bit/元素),
 * 解出来的 Δ 与真正落地的 Δ 不是同一个东西, 拿解算态的预测去报翻转率就是假账。 */
extern "C" int v41_gr_margin_predict(const float *dye, const float *drw, const int *dsel, const float *dC,
                                     const int *dsrc, const float *dx, int npair, int nu, int D, int n_expert,
                                     const float *dgate, int R, float *dm) {
    mg_ax_kernel<<<npair, 256>>>(dm, dx, dye, drw, dsel, dC, dsrc, nu, D, n_expert > 1 ? D : 0,
                                 dgate, R, (size_t)n_expert * D);
    GRCK(cudaGetLastError());
    GRCK(cudaDeviceSynchronize());
    return 0;
}

/* 解 §4.2。行序: [决策行 0..ndec) 在前, 约束行在后(与取料侧的置换同一套)。
 *   db[n]  决策行 = τ − m⁰(想推的量), 约束行 = 0
 *   dw[n]  行权重: 决策行 1(主动集里不参与的行传 0), 约束行 ρ
 *   dx[ne][D] 输出 Δ(偏离量, s = 1 + Δ); dm[n] 输出 A·Δ
 * 返回 0 成功; out_stat = {CG 末残差相对值, 迭代数}。 */
extern "C" int v41_gr_solve_margin_gpu(const float *dye, const float *drw, const int *dsel, const float *dC,
                                       const int *dsrc, const float *db, const float *dw, int n, int nu, int D,
                                       int n_expert, float lam, const float *dwe, const float *dgate, int R,
                                       int niter, float *dx, float *dm, float *out_stat) {
    const size_t xsz = (size_t)n_expert * D, nx = xsz * (size_t)R;
    /* ★进门先清粘性错误★: CUDA 的错误是粘在上下文上的, 不清掉的话本函数第一处 GRCK 会替
     * 上一步(门控上传、诊断里的临时缓冲…)背锅, 报出来的行号完全指错地方 —— 09-14 实撞。 */
    {
        const cudaError_t pre = cudaGetLastError();
        if (pre != cudaSuccess)
            fprintf(stderr, "★进解算前就有未清的 CUDA 错误: %s(R=%d n_expert=%d D=%d nx=%zu)★\n",
                    cudaGetErrorString(pre), R, n_expert, D, nx);
    }
    float *M = NULL, *ridge = NULL, *r = NULL, *z = NULL, *p = NULL, *hp = NULL, *rr = NULL, *mrow = NULL;
    double *dsum = NULL;
    GRCK(cudaMalloc((void **)&M, nx * 4));         GRCK(cudaMalloc((void **)&ridge, (size_t)D * 4));
    GRCK(cudaMalloc((void **)&r, nx * 4));         GRCK(cudaMalloc((void **)&z, nx * 4));
    GRCK(cudaMalloc((void **)&p, nx * 4));         GRCK(cudaMalloc((void **)&hp, nx * 4));
    GRCK(cudaMalloc((void **)&rr, (size_t)n * 4)); GRCK(cudaMalloc((void **)&mrow, (size_t)n * 4));
    GRCK(cudaMalloc((void **)&dsum, sizeof(double)));
    const dim3 sc((D + 255) / 256, n), se((D + 255) / 256, n_expert * R);
    /* ★n_expert == 1 = 共享模式★: 未知数只有 D 个(全体专家共用一组通道增益), 索引里的专家维塌掉。
     * 为什么要这条路: 每专家每通道各自独立时(1.97M 自由度), 最小范数解拟合的是【逐 token 的
     * 专家输出 ye】, 而 ye 在 5120 维里近似正交 ⇒ 换个位置投影归零, 泛化恒为 0(09-13/14 实测:
     * 样本 3 条→8 条, val 55.26% 一位小数不差)。共享之后解出来的必然是"对该通道所有 token
     * 一致的倾向", 才谈得上迁移。 */
    const int est = n_expert > 1 ? D : 0;
    const int nrb = (n + 255) / 256;
    const size_t nblk = (nx + 255) / 256;

    /* ① 数据对角 dd[e][d] = Σ_i w_i·a_i[k][d]² → ridge 尺度 + Jacobi 预条件 M(原地写回 M) */
    GRCK(cudaMemset(M, 0, nx * 4));
    mg_atr_kernel<<<sc, 256>>>(M, dw, dye, drw, dsel, dC, dsrc, n, nu, D, 1, est, dgate, R, xsz);
    GRCK(cudaGetLastError());
    mg_ridge_kernel<<<D, 256>>>(M, ridge, dwe, n_expert, D, lam, R);
    GRCK(cudaGetLastError());

    /* ② x = 0 起步 ⇒ 初始残差 r = Aᵀ W b */
    GRCK(cudaMemset(dx, 0, nx * 4));
    mg_rowscale<<<nrb, 256>>>(rr, db, dw, n);
    GRCK(cudaMemset(r, 0, nx * 4));
    mg_atr_kernel<<<sc, 256>>>(r, rr, dye, drw, dsel, dC, dsrc, n, nu, D, 0, est, dgate, R, xsz);
    GRCK(cudaGetLastError());
    mg_precond<<<nblk, 256>>>(z, r, M, nx);
    GRCK(cudaMemcpy(p, z, nx * 4, cudaMemcpyDeviceToDevice));
    double rz = mg_dot(dsum, r, z, nx);
    const double rz0 = rz;
    int it = 0;
    for (; it < niter && rz > 1e-14 * (rz0 > 0 ? rz0 : 1.0); it++) {
        /* Hp = Aᵀ W (A p) + ridge·p */
        mg_ax_kernel<<<n, 256>>>(mrow, p, dye, drw, dsel, dC, dsrc, nu, D, est, dgate, R, xsz);
        mg_rowscale<<<nrb, 256>>>(rr, mrow, dw, n);
        GRCK(cudaMemset(hp, 0, nx * 4));
        mg_atr_kernel<<<sc, 256>>>(hp, rr, dye, drw, dsel, dC, dsrc, n, nu, D, 0, est, dgate, R, xsz);
        mg_ridge_add<<<se, 256>>>(hp, p, ridge, dwe, n_expert, D);
        GRCK(cudaGetLastError());
        const double php = mg_dot(dsum, p, hp, nx);
        if (!(php > 0.0)) break;   /* 数值上不再正定 = 已经收敛到头, 再走就是噪声 */
        const float a = (float)(rz / php);
        mg_axpy<<<nblk, 256>>>(dx, p, a, nx);
        mg_axpy<<<nblk, 256>>>(r, hp, -a, nx);
        mg_precond<<<nblk, 256>>>(z, r, M, nx);
        const double rz_new = mg_dot(dsum, r, z, nx);
        mg_pz<<<nblk, 256>>>(p, z, (float)(rz_new / rz), nx);
        rz = rz_new;
    }
    mg_ax_kernel<<<n, 256>>>(dm, dx, dye, drw, dsel, dC, dsrc, nu, D, est, dgate, R, xsz);
    GRCK(cudaGetLastError());
    GRCK(cudaDeviceSynchronize());
    if (out_stat) { out_stat[0] = (float)(rz0 > 0 ? sqrt(rz / rz0) : 0.0); out_stat[1] = (float)it; }
    cudaFree(M); cudaFree(ridge); cudaFree(r); cudaFree(z); cudaFree(p); cudaFree(hp);
    cudaFree(rr); cudaFree(mrow); cudaFree(dsum);
    return 0;
}

/* ★诊断: 决策方程之间的 Gram 矩阵★(2026-09-14, back.md §4.8 诊断 A)。
 * 为什么要它: 解是最小范数解, 恒落在 span{A_1..A_n}(训练决策方程张成的子空间)里, 所以对一个
 * 没见过的决策点 j, 效果 Δm_j = Σ_i β_i·<A_j, A_i> —— ★A_j 与训练集的 A 近正交时, 泛化必然为 0★,
 * 与 λ、与样本量、与解算精度全都无关。09-14 实测(8 条样本 12 个候选)val 恒 55.26% 一位小数不差,
 * 就是这个形状。把 Gram 算出来, "泛化上限"就从猜变成一个能看的数(投影占比, 主机侧 Cholesky)。
 *
 * A_i 是 [n_expert][D] 空间里的稀疏向量: A_i[e][d] = Σ_{k: sel[i][k]=e} rw[i][k]·ye[i][k][d]·C_i[d]。
 * G[i][j] = Σ_d C_i[d]C_j[d] · Σ_{k,k': 同专家} rw·ye·rw'·ye'。一个 block 一对, 只算上三角。 */
__global__ static void mg_gram_kernel(float *G, const float *ye, const float *rw, const int *sel,
                                      const float *C, const int *src, int np, int nu, int D) {
    const int i = blockIdx.x, j = blockIdx.y;
    if (j < i) return;
    __shared__ double sh[256];
    const int ri = src[i], rj = src[j];
    const float *ci = C + (size_t)i * D, *cj = C + (size_t)j * D;
    double acc = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        double a = 0.0;
        for (int k = 0; k < nu; k++) {
            const int e = sel[(size_t)ri * nu + k];
            if (e < 0) continue;
            const double vi = (double)rw[(size_t)ri * nu + k] * ye[((size_t)ri * nu + k) * (size_t)D + d];
            for (int k2 = 0; k2 < nu; k2++) {
                if (sel[(size_t)rj * nu + k2] != e) continue;
                a += vi * rw[(size_t)rj * nu + k2] * ye[((size_t)rj * nu + k2) * (size_t)D + d];
            }
        }
        acc += a * (double)ci[d] * cj[d];
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) { G[(size_t)i * np + j] = (float)sh[0]; G[(size_t)j * np + i] = (float)sh[0]; }
}

extern "C" int v41_gr_margin_gram(const float *dye, const float *drw, const int *dsel, const float *dC,
                                  const int *dsrc, int np, int nu, int D, float *dG) {
    mg_gram_kernel<<<dim3(np, np), 256>>>(dG, dye, drw, dsel, dC, dsrc, np, nu, D);
    GRCK(cudaGetLastError());
    GRCK(cudaDeviceSynchronize());
    return 0;
}
