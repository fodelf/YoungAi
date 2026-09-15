/* cuda_v41_nvfp4.inc.cu — ds4_cuda.cu 分片: V4.1 预填稠密 GEMM 走 NVFP4 张量核(2026-09-15, speed.md S1)。
 *
 * 【为什么是 NVFP4 而不是我们盘上的 MXFP4】S0 实测(gguf-tools/bench/gemm_fp4_ceiling.cu, 09-15):
 * GB10 上 cuBLASLt 对 MXFP4(ue8m0/32)与"FP4 权重×FP8 激活"都**没有算法**; 唯一能走张量核的是
 * NVFP4 = E2M1 + 每 16 个元素一个 E4M3 缩放, 实测 284~356 TFLOPS(BF16 81~91, FP8 76~188)。
 *
 * 【盘上格式不动】fp4x32(17 B/32 = 16 B nibble + 1 B ue8m0)是解码 GEMV 的口径(4.25 bpw 比 NVFP4 的
 * 4.5 省字节), 一个字节不改。本分片只在**预填暂存**里转: ★nibble 字节两种格式完全相同★, 转换 = 去交织
 * (16 B 搬到 nibble 平面) + 把那个 2 的幂写成两个 e4m3 缩放字节。暂存字节 18 B/32 元素, 比原来的
 * f16 路(64 B/32)少 3.5×, 加上 GEMM 快 3.5× —— 两头都赚。
 *
 * 【恒等的边界】e4m3 只能精确表示 2^-9..2^8 的 2 的幂。盘上实扫(dump_gguf_meta --fp4scan, 09-15,
 * 2.452 亿块)越界的只有 14013 块 = 0.0057%, 全在下界侧(2^-12..2^-10)且值 ≤0.0059 —— 近零块。
 * 这些块按 scale=2^-9 重新舍入 nibble(不是静默饱和: 值除以 2^(p+9) 后重新走 E2M1 最近舍入),
 * 并计数到 g_v41_nvfp4_clamped, --v41-prof 时打印。其余 99.994% 逐位恒等。
 *
 * 【缩放张量布局】128×4 swizzle, 09-15 用 gemm_fp4_ceiling --check 逐格验过(7/7):
 * off(r,c) = ((r/128)*ceil(nkb/4) + c/4)*512 + (r%32)*16 + ((r%128)/32)*4 + c%4
 * 猜错不报错只出假数, 所以这个公式与探针里那份必须同式(改一处就要重跑 --check)。 */

#define V41_NVFP4_BLK 16u          /* NVFP4 缩放块 = 16 个元素(MXFP4 是 32, 一块拆两个) */
#define V41_NVFP4_MAX_CACHE 128    /* (m,n,k) → 算法缓存槽。★09-15 从 24 加到 128★: 段 5 的专家路按
                                    * "n 对齐到 2 的幂"分桶(8..2048 共 9 档)× 两种矩阵形状 = 18 种,
                                    * 加上骨架那十来种就超了 24, 报"算法缓存满"直接停车。一个槽才几十字节。 */

static v41_scratch g_v41_wnib, g_v41_wsc, g_v41_xnib, g_v41_xsc;
static unsigned long long *g_v41_nvfp4_clamped = NULL;   /* 设备计数器(越界块数) */

/* 2^p → e4m3 字节。正规 p∈[-6,8]: (p+7)<<3; 次正规 p∈[-9,-7]: 尾数 1/2/4(值 = mant·2^-9)。
 * 与 ds4_fp8.h 的 ds4_e4m3fn_value 表逐值对得上(exp_scale[exp] = 2^(exp-7), 尾数 0)。 */
__device__ __forceinline__ static uint8_t v41_pow2_to_e4m3(int p) {
    if (p >= -6) return (uint8_t)(((p > 8 ? 8 : p) + 7) << 3);
    if (p == -7) return 0x04u;
    if (p == -8) return 0x02u;
    return 0x01u;   /* p <= -9 一律 2^-9(下界), 调用方负责把 nibble 按差额缩小 */
}
__device__ __forceinline__ static size_t v41_nvfp4_scale_off(uint32_t r, uint32_t c, uint32_t nkb) {
    const uint32_t tiles_c = (nkb + 3u) / 4u;
    return (size_t)((r / 128u) * tiles_c + c / 4u) * 512u + (size_t)(r % 32u) * 16u + (size_t)((r % 128u) / 32u) * 4u + (size_t)(c % 4u);
}

/* 权重: MXFP4 块(17 B) → NVFP4。一线程一块。row = blk / kb32, kb32 = blk % kb32。 */
__global__ static void v41_mxfp4_to_nvfp4_kernel(uint8_t *nib, uint8_t *sc, const uint8_t *w,
                                                 uint32_t rows, uint32_t kb32, uint32_t nkb,
                                                 unsigned long long *clamped) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (uint64_t)rows * kb32) return;
    const uint32_t r = (uint32_t)(b / kb32), c32 = (uint32_t)(b % kb32);
    const uint8_t *p = w + b * 17u;
    const int e = (int)p[16], pw = e - 127;          /* ue8m0: 值 = 2^(e-127) */
    uint8_t *dst = nib + (uint64_t)r * (kb32 * 16u) + (uint64_t)c32 * 16u;
    if (pw >= -9 && pw <= 8) {                        /* 恒等路: nibble 原样搬 */
        #pragma unroll
        for (int j = 0; j < 16; j++) dst[j] = p[j];
    } else {                                          /* 越界: 缩放钉在边界, nibble 按差额重量化 */
        const float f = exp2f((float)(pw - (pw < -9 ? -9 : 8)));
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const float lo = ds4_fp4_nibble_to_f32(p[j] & 0x0Fu) * f;
            const float hi = ds4_fp4_nibble_to_f32(p[j] >> 4) * f;
            dst[j] = (uint8_t)(ds4_fp4_f32_to_nibble(lo) | (ds4_fp4_f32_to_nibble(hi) << 4));
        }
        if (clamped) atomicAdd(clamped, 1ull);
    }
    const uint8_t sb = v41_pow2_to_e4m3(pw);          /* 一个 32 块 = 两个 16 块, 同一个缩放 */
    sc[v41_nvfp4_scale_off(r, c32 * 2u, nkb)] = sb;
    sc[v41_nvfp4_scale_off(r, c32 * 2u + 1u, nkb)] = sb;
}

/* 激活: f32 → NVFP4(每 16 个一块, e4m3 缩放)。一线程一块, 照官方 fp4_quant_kernel 的 e4m3 分支:
 * amax 下限 6·2^-9(全零块的缩放也不许是 0), scale = e4m3(amax/6), q = E2M1(clamp(x/s, ±6))。 */
__global__ static void v41_x_to_nvfp4_kernel(uint8_t *nib, uint8_t *sc, const float *x,
                                             uint32_t rows, uint32_t dim, uint32_t nkb, uint32_t x_stride) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (uint64_t)rows * nkb) return;
    const uint32_t r = (uint32_t)(b / nkb), c = (uint32_t)(b % nkb);
    const float *src = x + (uint64_t)r * x_stride + (uint64_t)c * V41_NVFP4_BLK;   /* x_stride: 分组路一行里只取本组那一段 */
    float amax = 6.0f * exp2f(-9.0f);
    #pragma unroll
    for (int j = 0; j < (int)V41_NVFP4_BLK; j++) amax = fmaxf(amax, fabsf(src[j]));
    const float s = ds4_e4m3fn_round(amax / 6.0f), inv = s > 0.0f ? 1.0f / s : 0.0f;
    uint8_t *dst = nib + (uint64_t)r * (dim / 2u) + (uint64_t)c * (V41_NVFP4_BLK / 2u);
    #pragma unroll
    for (int j = 0; j < (int)V41_NVFP4_BLK / 2; j++) {
        const float a = fminf(fmaxf(src[2 * j] * inv, -6.0f), 6.0f);
        const float bb = fminf(fmaxf(src[2 * j + 1] * inv, -6.0f), 6.0f);
        dst[j] = (uint8_t)(ds4_fp4_f32_to_nibble(a) | (ds4_fp4_f32_to_nibble(bb) << 4));
    }
    /* e4m3 值 → 字节: 走 ds4_fp8.h 的幅值表反查(表里是精确值, 相等比较安全) */
    uint8_t sb = 0;
    for (int i = 1; i < 127; i++) if (ds4_e4m3fn_value(i) == s) { sb = (uint8_t)i; break; }
    sc[v41_nvfp4_scale_off(r, c, nkb)] = sb;
}

/* cuBLASLt: D[m=out_dim, n=n_tok] 列主序(= 行主序 [n_tok][out_dim]) = Aᵀ·B, A/B 都按 K 连续。 */
typedef struct { int m, n, k, ldd, valid; cublasLtMatmulAlgo_t algo; } v41_lt_cache;
static v41_lt_cache g_v41_lt[V41_NVFP4_MAX_CACHE];
static int g_v41_lt_n = 0;
static void *g_v41_lt_ws = NULL;
static const size_t V41_LT_WS = 64u << 20;

static int v41_gemm_nvfp4(const uint8_t *Anib, const uint8_t *Asc, const uint8_t *Bnib, const uint8_t *Bsc,
                          float *D, int m, int n, int k, int ldd, const char *what) {
    if (!g_v41_lt_ws && cudaMalloc(&g_v41_lt_ws, V41_LT_WS) != cudaSuccess) {
        (void)cudaGetLastError(); fprintf(stderr, "ds4: [v41] NVFP4 工作区分配失败\n"); return 0;
    }
    /* cublasHandle_t 与 cublasLtHandle_t 是同一个对象, 官方文档允许直接转型 —— 不另建 Lt 句柄,
     * 免得两套句柄各挂一条流(V4.1 全链一条流, 见 v41_cublas_stream 的注释) */
    cublasLtHandle_t lt = (cublasLtHandle_t)g_cublas;
    cublasLtMatmulDesc_t op = NULL; cublasLtMatrixLayout_t la = NULL, lb = NULL, ld = NULL;
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    int rc = 0, ok = 1, slot = -1;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return 0;
    ok &= cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &Asc, sizeof Asc) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &Bsc, sizeof Bsc) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof sm) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof sm) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, (uint64_t)k, (uint64_t)m, (int64_t)k) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, (uint64_t)k, (uint64_t)n, (int64_t)k) == CUBLAS_STATUS_SUCCESS;
    ok &= cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, (uint64_t)m, (uint64_t)n, (int64_t)ldd) == CUBLAS_STATUS_SUCCESS;
    if (!ok) fprintf(stderr, "ds4: [v41] %s NVFP4 描述符建失败\n", what);
    if (ok) {
        for (int i = 0; i < g_v41_lt_n; i++)
            if (g_v41_lt[i].m == m && g_v41_lt[i].n == n && g_v41_lt[i].k == k && g_v41_lt[i].ldd == ldd) { slot = i; break; }
    }
    if (ok && slot < 0) {   /* 首次见这个形状: 问一次启发式并缓存(每层十来种形状, 不会打满) */
        cublasLtMatmulPreference_t pref = NULL;
        cublasLtMatmulHeuristicResult_t heur[1]; int nh = 0;
        size_t ws = V41_LT_WS;
        if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
            (void)cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
            if (cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, heur, &nh) != CUBLAS_STATUS_SUCCESS) nh = 0;
            cublasLtMatmulPreferenceDestroy(pref);
        }
        if (nh == 0) {
            /* 不退 f16: 悄悄退回去就永远不知道哪条路在跑(禁兜底铁律), 而且速度读数会成谜 */
            fprintf(stderr, "ds4: [v41] %s NVFP4 无可用算法 (m=%d n=%d k=%d) -- aborting\n", what, m, n, k);
            ok = 0;
        } else if (g_v41_lt_n >= V41_NVFP4_MAX_CACHE) {
            fprintf(stderr, "ds4: [v41] NVFP4 算法缓存满(%d 种形状) -- aborting\n", g_v41_lt_n); ok = 0;
        } else {
            slot = g_v41_lt_n++;
            g_v41_lt[slot].m = m; g_v41_lt[slot].n = n; g_v41_lt[slot].k = k; g_v41_lt[slot].ldd = ldd;
            g_v41_lt[slot].algo = heur[0].algo; g_v41_lt[slot].valid = 1;
        }
    }
    if (ok) {
        const float alpha = 1.0f, beta = 0.0f;
        rc = cublas_ok(cublasLtMatmul(lt, op, &alpha, Anib, la, Bnib, lb, &beta, D, ld, D, ld,
                                      &g_v41_lt[slot].algo, g_v41_lt_ws, V41_LT_WS, v41_cublas_stream()), what);
    }
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (ld) cublasLtMatrixLayoutDestroy(ld);
    if (op) cublasLtMatmulDescDestroy(op);
    return rc;
}

/* 稠密 fp4x32 权重 × f32 激活 → f32 出口, 全程 NVFP4 张量核。in_dim 必须 32 的倍数(盘上格式保证)。 */
static int v41_matmul_nvfp4(const void *model_map, uint64_t model_size, uint64_t off,
                            uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                            uint32_t n_tok, const char *what) {
    if ((in_dim % 32u) != 0u) return 0;
    const uint32_t kb32 = (uint32_t)(in_dim / 32u), nkb = (uint32_t)(in_dim / V41_NVFP4_BLK);
    const uint64_t nblk = out_dim * kb32, wbytes = nblk * 17u;
    if (off > model_size || wbytes > model_size - off) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, off, wbytes, what);
    if (!w) return 0;
    /* 缩放张量按 128×4 swizzle 补齐(少分配不报错只越界读) */
    const uint64_t wsc_n = ((out_dim + 127u) / 128u * 128u) * ((nkb + 3u) / 4u * 4u);
    const uint64_t xsc_n = ((n_tok + 127u) / 128u * 128u) * ((nkb + 3u) / 4u * 4u);
    uint8_t *wnib = (uint8_t *)v41_grow(&g_v41_wnib, out_dim * (in_dim / 2u), "v41 nvfp4 权重");
    uint8_t *wsc  = (uint8_t *)v41_grow(&g_v41_wsc, wsc_n, "v41 nvfp4 权重缩放");
    uint8_t *xnib = (uint8_t *)v41_grow(&g_v41_xnib, (uint64_t)n_tok * (in_dim / 2u), "v41 nvfp4 激活");
    uint8_t *xsc  = (uint8_t *)v41_grow(&g_v41_xsc, xsc_n, "v41 nvfp4 激活缩放");
    if (!wnib || !wsc || !xnib || !xsc) return 0;
    if (!g_v41_nvfp4_clamped) (void)cudaMalloc(&g_v41_nvfp4_clamped, sizeof(unsigned long long));
    /* 补齐区必须清零: 那里的字节会被张量核读到(只影响补齐行的无效输出, 但脏值可能是 NaN) */
    (void)cudaMemsetAsync(wsc, 0, wsc_n, g_cur_stream);
    (void)cudaMemsetAsync(xsc, 0, xsc_n, g_cur_stream);
    v41_mxfp4_to_nvfp4_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, g_cur_stream>>>(
        wnib, wsc, w, (uint32_t)out_dim, kb32, nkb, g_v41_nvfp4_clamped);
    if (!cuda_ok(cudaGetLastError(), "v41 mxfp4→nvfp4")) return 0;
    const uint64_t xblk = (uint64_t)n_tok * nkb;
    v41_x_to_nvfp4_kernel<<<(unsigned)((xblk + 255) / 256), 256, 0, g_cur_stream>>>(
        xnib, xsc, x, n_tok, (uint32_t)in_dim, nkb, (uint32_t)in_dim);
    if (!cuda_ok(cudaGetLastError(), "v41 x→nvfp4")) return 0;
    return v41_gemm_nvfp4(wnib, wsc, xnib, xsc, out, (int)out_dim, (int)n_tok, (int)in_dim, (int)out_dim, what);
}

/* ★2026-09-15 clear.md 判负存档: wo_a(块对角 8 组)走 NVFP4★
 * 写过一版 v41_grouped_matmul_nvfp4(权重整份转 NVFP4, 缩放张量按 128 行 tile 切组 —— rank=1024 是
 * 128 的倍数所以切得动; 激活按组各转一份, 输出靠 cuBLASLt 的 ldd 写列段)。同机器状态 A/B:
 *   f16 老路 20.8~21.1 s / PPL 15.8403 ; NVFP4 20.6~20.8 s / PPL 16.4556 —— **速度一样, 质量退 3.9%**。
 * wo_a 不是预填瓶颈, 降精度一毫秒都换不回来, 已删。现在这条路走 bf16 张量核(cuda_v41_1.inc.cu),
 * 对 FP4 权重逐位无损。★要重做请先想清楚拿什么换那 3.9%★。
 * 顺带留下的两个参数不要删: v41_x_to_nvfp4_kernel 的 x_stride 与 v41_gemm_nvfp4 的 ldd —— 专家路
 * 也在用它们表达"一行里只取一段 / 往列段里写"。 */
