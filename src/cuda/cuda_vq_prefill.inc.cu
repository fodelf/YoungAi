/* cuda_vq_prefill.inc.cu — VQ 专家的批量(prefill) GEMM 路(2026-09-06)。
 *
 * 为什么: 原 prefill 路(已删)每个 256 token 块把本层全部活跃专家(≈256 个)dequant 成 f16 落
 * 12.9 GB scratch, 再用逐 (token,pick) 的 warp 核算 —— 8192 token 一趟 dequant 60 万次(63 s)、
 * gateup/down 核 78 s, 算力只跑到 44 GFLOPS; 合起来 prefill 20 t/s, 比 decode 还慢。
 * 这里: (token,pick) 对按专家排序 → 每个专家只 dequant 一次(3 个矩阵 42 MB, 用完即弃, 不落
 * 缓存)→ 该专家名下的 n_e 行激活做两发 cuBLAS f16 GEMM(gate+up 拼成一个 [2·MID×IN] 权重
 * 一发, down 一发)→ 最后按 token 以固定 pick 序加权求和(定序 ⇒ 同输入可复现)。scratch 不到
 * 1 GB。每块固定成本 = 43 层 × 256 专家 × 3 次 dequant ≈ 3.5 s, 所以 prefill 块越大越省:
 * 块 4096 时折合 0.85 ms/token(块默认值见 core_kv.c)。
 * 数值: 激活 f32→f16、h f32→f16(与 f16 骨架 prefill 同口径); decode 路(fused2)读 f32 激活。
 * 两路不逐位同, 判决走批路五指标(kernel_parity_spark.sh … prefill)。
 * 改了会怎样: 把 dequant 改回"全层落缓存跨块复用"要 43 层 × 12.9 GB, 单机放不下, 别走回头路;
 * reduce 若改成 atomicAdd 会回到 08-22 删掉的"同 prompt 两次跑不同"。 */

/* 权重侧反修(--zchain 目录里的 gr_Lnn.bin): [layer] → 设备上的 s[n_expert][OUT] 缩放因子, NULL = 该层不挂。
 * 只作用在 down 的行增益上(行 = 输出通道)。挂了就 100% 生效, 不做任何静默回退。 */
static float *g_v41_gr[64];

static struct {
    __half  *xs;   uint64_t xs_cap;    /* 排序后的激活 [nvalid][IN] f16 */
    float   *ys;   uint64_t ys_cap;    /* 排序后的 down 输出 [nvalid][OUT] f32 */
    int32_t *perm; uint64_t perm_cap;  /* 排序位置 → pair 编号 */
    int32_t *inv;  uint64_t inv_cap;   /* pair 编号 → 排序位置(-1=无效 pick) */
} g_vqp;

/* 侧流流水(pf3): 专家 i 走 lane i%N, lane 内串行(dequant→GEMM→swiglu→GEMM), lane 间并行 ⇒
 * 带宽型 dequant(每专家 42 MB f16 落盘, 180 GB/s)与算力型 GEMM 重叠; 小 n 的专家 GEMM
 * (m=4096 一段只有 32 个 CTA, 填不满 48 SM)也能几发并跑。每 lane 自带 cuBLAS 句柄: 同一句柄
 * 换流会让各流共用一份 workspace(cuBLAS 文档明令禁止)。同步只靠事件: 主流 gather 后记 start,
 * lane 等 start; 末尾主流等全部 lane 的 done 再 reduce —— lane 是 NonBlocking 流, 不与
 * 默认流隐式同步。结果逐位不变: 每专家写自己的 ys 行段, reduce 定序。 */
#define VQP_NLANE 4u
typedef struct {
    cudaStream_t st; cublasHandle_t bl; cudaEvent_t done;
    float   *gu;   uint64_t gu_cap;    /* 单专家 gate|up 输出 [ne_max][2·MID] f32 */
    __half  *h;    uint64_t h_cap;     /* 单专家 SwiGLU 中间 [ne_max][MID] f16 */
    __half  *wgu;  uint64_t wgu_cap;   /* 单专家权重 gate|up [2·MID][IN] f16 */
    __half  *wd;   uint64_t wd_cap;    /* 单专家权重 down [OUT][MID] f16 */
} vqp_lane;
/* 反修取料(2026-09-13): 上一次 prefill MoE 的形状 + 展开缓冲。形状用来校验取料方要的层对不对得上。 */
static uint32_t g_vqp_last_tok = 0, g_vqp_last_used = 0, g_vqp_last_out = 0;
static float *g_vqp_cap = NULL; static uint64_t g_vqp_cap_n = 0;

static vqp_lane g_vqp_lane[VQP_NLANE];
static cudaEvent_t g_vqp_ev_start = NULL;
static int g_vqp_lanes_ready = 0;

static int vqp_lanes_init(void) {
    if (g_vqp_lanes_ready) return 1;
    cublasMath_t mm = CUBLAS_DEFAULT_MATH;
    (void)cublasGetMathMode(g_cublas, &mm);   /* 与主句柄同数学模式(quality 开关) */
    if (cudaEventCreateWithFlags(&g_vqp_ev_start, cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    for (uint32_t s = 0; s < VQP_NLANE; s++) {
        vqp_lane *L = &g_vqp_lane[s];
        if (cudaStreamCreateWithFlags(&L->st, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&L->done, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError(); return 0;
        }
        if (!cublas_ok(cublasCreate(&L->bl), "vq prefill lane handle")) return 0;
        (void)cublasSetMathMode(L->bl, mm);
        if (!cublas_ok(cublasSetStream(L->bl, L->st), "vq prefill lane stream")) return 0;
    }
    g_vqp_lanes_ready = 1;
    return 1;
}

static int vqp_grow(void **p, uint64_t *cap, uint64_t need, size_t elem, const char *what) {
    if (need <= *cap) return 1;
    (void)cudaDeviceSynchronize();   /* 旧块可能仍被任一流的在飞 kernel 读; 增长只发生在头几层 */
    if (*p) (void)cudaFree(*p);
    *p = NULL; *cap = 0;
    if (cudaMalloc(p, need * elem) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [vq-prefill] %s 分配失败 (%.1f MB)\n", what, (double)need * elem / 1048576.0);
        return 0;
    }
    *cap = need;
    return 1;
}

/* 排序位置 i 取 pair perm[i] 所属 token 的激活行, f32→f16 */
__global__ static void vqp_gather_kernel(__half *xs, const float *x, const int32_t *perm,
                                         uint32_t n_expert, uint32_t IN) {
    const uint32_t i = blockIdx.x;
    const uint32_t t = (uint32_t)perm[i] / n_expert;
    const float *src = x + (uint64_t)t * IN;
    __half *dst = xs + (uint64_t)i * IN;
    for (uint32_t k = threadIdx.x; k < IN; k += blockDim.x) dst[k] = __float2half(src[k]);
}

/* h = silu(clamp_hi(g)) · clamp(u): clamp 语义同 decode 路(gate 只截上界, up 双向截) */
__global__ static void vqp_swiglu_kernel(__half *h, const float *gu, uint32_t MID, float clamp) {
    const uint32_t row = blockIdx.x;
    const float *g = gu + (uint64_t)row * 2u * MID, *u = g + MID;
    __half *o = h + (uint64_t)row * MID;
    for (uint32_t m = threadIdx.x; m < MID; m += blockDim.x) {
        float gv = g[m], uv = u[m];
        if (clamp > 0.0f) {
            if (gv > clamp) gv = clamp;
            if (uv > clamp) uv = clamp;
            if (uv < -clamp) uv = -clamp;
        }
        o[m] = __float2half((gv / (1.0f + __expf(-gv))) * uv);
    }
}

/* out[t][o] = Σ_pk w[t][pk] · ys[inv[t·n_expert+pk]][o], pick 序固定 ⇒ 可复现 */
__global__ static void vqp_reduce_kernel(float *out, const float *ys, const int32_t *inv, const float *rw,
                                         uint32_t n_expert, uint32_t OUT) {
    const uint32_t t = blockIdx.y;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float s = 0.0f;
    for (uint32_t pk = 0; pk < n_expert; pk++) {
        const int32_t i = inv[(uint64_t)t * n_expert + pk];
        if (i < 0) continue;
        s += rw[(uint64_t)t * n_expert + pk] * ys[(uint64_t)i * OUT + o];
    }
    out[(uint64_t)t * OUT + o] = s;
}

/* 载荷头缓存(每层一次): 从显存里的 blob 抄 offset 表 + 每槽 16 B 头到主机, 校验后缓存 dim/nc/nbit。
 * 为什么: blob 的 mmap 原件在启动拷进显存后被 madvise(DONTNEED), 主机再碰一个头就是一次缺页
 * 回读 SSD; 每专家 3 个头散在 78 GB 里 ⇒ 每层 768 次缺页, 剖面里表现为专家与专家之间 8~35 ms
 * 的 GPU 空转(4096 块: GPU 忙 10.8 s / 空转 4.9 s, pf1)。校验规则与 cuda_vq_pay_hdr 同。 */
typedef struct { uint64_t off; uint32_t dim, nc, nbit; } vqp_slot_hdr;   /* off=0 ⇒ 槽缺席 */
static vqp_slot_hdr *g_vqp_hdr[64];        /* [layer] → [e*3+which] */
static uint32_t *g_vqp_hdr_dev = NULL;     /* 抄头暂存 [n_total*3][6] u32: off(2) magic dim|nc rows cols */
static uint64_t g_vqp_hdr_dev_n = 0;

__global__ static void vqp_copy_hdr_kernel(uint32_t *dst, const uint8_t *blob, uint32_t n_total) {
    const uint32_t e = blockIdx.x, w = threadIdx.x;
    if (e >= n_total || w >= 3u) return;
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + w) * 8, 8);
    uint32_t *d = dst + ((size_t)e * 3 + w) * 6;
    memcpy(d, &off, 8);
    if (off) memcpy(d + 2, blob + off, 16);
    else d[2] = d[3] = d[4] = d[5] = 0;
}

static int vqp_hdr_build(uint32_t layer, const uint8_t *blob, uint32_t n_total,
                         uint32_t IN, uint32_t MID, uint32_t OUT) {
    if (layer >= 64u) return 0;
    if (g_vqp_hdr[layer]) return 1;
    const uint64_t n = (uint64_t)n_total * 3u;
    if (!vqp_grow((void **)&g_vqp_hdr_dev, &g_vqp_hdr_dev_n, n * 6u, sizeof(uint32_t), "hdr")) return 0;
    vqp_copy_hdr_kernel<<<n_total, 4, 0, g_cur_stream>>>(g_vqp_hdr_dev, blob, n_total);
    if (!cuda_ok(cudaGetLastError(), "vq prefill hdr copy launch")) return 0;
    uint32_t *raw = (uint32_t *)malloc((size_t)n * 6u * sizeof(uint32_t));
    vqp_slot_hdr *tab = (vqp_slot_hdr *)calloc((size_t)n, sizeof(vqp_slot_hdr));
    if (!raw || !tab ||
        cudaMemcpy(raw, g_vqp_hdr_dev, (size_t)n * 6u * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError(); free(raw); free(tab); return 0;
    }
    int bad = 0;
    for (uint64_t k = 0; k < n; k++) {
        const uint32_t *d = raw + k * 6u;
        uint64_t off; memcpy(&off, d, 8);
        tab[k].off = off;
        if (!off) continue;
        const uint32_t which = (uint32_t)(k % 3u), e = (uint32_t)(k / 3u);
        const uint32_t exp_rows = which == 2u ? OUT : MID, exp_cols = which == 2u ? MID : IN;
        const uint32_t d16 = d[3] & 0xFFFFu, n16 = d[3] >> 16;
        if (d[2] != DS4VQ_MAT_MAGIC || d[4] != exp_rows || d[5] != exp_cols) {
            fprintf(stderr, "ds4: [vq-prefill] L%u e=%u which=%u 头错(magic %08x %u×%u, 期 %u×%u) -- aborting\n",
                    layer, e, which, d[2], d[4], d[5], exp_rows, exp_cols);
            bad = 1; break;
        }
        /* 48 KB 是 fused2(dim4) 把码本搬 shared 的上限; 本路的 vq_dequant_kernel 码本走全局读, 不受限。
         * V4.1 dim8×nc4096 码本 64 KB 正好踩线(2026-09-12), 只对 dim==4 仍按 fused2 口径把关。 */
        if (d16 == 4u && (size_t)n16 * d16 * 2u > 48u * 1024u) {
            fprintf(stderr, "ds4: [vq-prefill] L%u e=%u 码本 %u×%u 超 shared 上限 48KB -- aborting\n", layer, e, n16, d16);
            bad = 1; break;
        }
        uint32_t nb = 0; while ((1u << nb) < n16) nb++; if (nb < 1u) nb = 1u;
        tab[k].dim = d16; tab[k].nc = n16; tab[k].nbit = nb;
    }
    free(raw);
    if (bad) { free(tab); return 0; }
    g_vqp_hdr[layer] = tab;
    return 1;
}

/* C[n][m] = B[n][k] · A[m][k]^T (行主序视角), f16 输入 f32 累加, 与稠密 f16 路同一 cuBLAS 口径 */
static int vqp_gemm(cublasHandle_t bl, const __half *A, int lda, const __half *B, int ldb, float *C, int ldc,
                    int m, int n, int k, const char *what) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(bl, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                                     A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &beta,
                                     C, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, what);
}

static int cuda_vq_moe_prefill_gemm(
        ds4_gpu_tensor *out, const uint8_t *blob,
        const void *model_map, uint64_t down_offset, uint64_t down_expert_bytes,
        uint32_t IN, uint32_t MID, uint32_t OUT,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens) {
    if (!g_cublas_ready) { fprintf(stderr, "ds4: [vq-prefill] cuBLAS 未就绪 (L%u)\n", layer_index); return 0; }
    if (!vqp_lanes_init()) { fprintf(stderr, "ds4: [vq-prefill] 侧流初始化失败 (L%u)\n", layer_index); return 0; }
    if (!vqp_hdr_build(layer_index, blob, n_total_expert, IN, MID, OUT)) return 0;
    const vqp_slot_hdr *tab = g_vqp_hdr[layer_index];
    const uint64_t npair = (uint64_t)n_tokens * n_expert;
    int32_t *sel_h = (int32_t *)malloc(npair * sizeof(int32_t));
    int32_t *perm_h = (int32_t *)malloc(npair * sizeof(int32_t));
    int32_t *inv_h = (int32_t *)malloc(npair * sizeof(int32_t));
    uint32_t *cnt = (uint32_t *)calloc((size_t)n_total_expert, sizeof(uint32_t));
    uint32_t *off = (uint32_t *)malloc(((size_t)n_total_expert + 1u) * sizeof(uint32_t));
    uint32_t *cur = (uint32_t *)malloc((size_t)n_total_expert * sizeof(uint32_t));
    int ok = 0;
    do {
        if (!sel_h || !perm_h || !inv_h || !cnt || !off || !cur) break;
        /* selected 取回主机做计数排序(每层一次 D2H, prefill 非捕获态, 老路同款) */
        if (cudaMemcpy(sel_h, selected->ptr, npair * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void)cudaGetLastError(); break;
        }
        for (uint64_t k = 0; k < npair; k++) {
            const int32_t e = sel_h[k];
            if (e >= 0 && (uint32_t)e < n_total_expert) cnt[e]++;
        }
        off[0] = 0;
        uint32_t ne_max = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            off[e + 1] = off[e] + cnt[e];
            if (cnt[e] > ne_max) ne_max = cnt[e];
        }
        const uint32_t nvalid = off[n_total_expert];
        if (nvalid == 0) {   /* 全是空槽: routed 输出为 0 */
            ok = (cudaMemsetAsync(out->ptr, 0, (size_t)n_tokens * OUT * sizeof(float), g_cur_stream) == cudaSuccess);
            break;
        }
        memcpy(cur, off, (size_t)n_total_expert * sizeof(uint32_t));
        for (uint64_t k = 0; k < npair; k++) {   /* 稳定: 同专家内保持 token 序 */
            const int32_t e = sel_h[k];
            if (e < 0 || (uint32_t)e >= n_total_expert) { inv_h[k] = -1; continue; }
            const uint32_t pos = cur[e]++;
            perm_h[pos] = (int32_t)k;
            inv_h[k] = (int32_t)pos;
        }
        if (!vqp_grow((void **)&g_vqp.xs, &g_vqp.xs_cap, (uint64_t)nvalid * IN, sizeof(__half), "xs") ||
            !vqp_grow((void **)&g_vqp.ys, &g_vqp.ys_cap, (uint64_t)nvalid * OUT, sizeof(float), "ys") ||
            !vqp_grow((void **)&g_vqp.perm, &g_vqp.perm_cap, nvalid, sizeof(int32_t), "perm") ||
            !vqp_grow((void **)&g_vqp.inv, &g_vqp.inv_cap, npair, sizeof(int32_t), "inv")) break;
        int grown = 1;
        for (uint32_t s = 0; s < VQP_NLANE && grown; s++) {
            vqp_lane *L = &g_vqp_lane[s];
            grown = vqp_grow((void **)&L->gu, &L->gu_cap, (uint64_t)ne_max * 2u * MID, sizeof(float), "gu") &&
                    vqp_grow((void **)&L->h, &L->h_cap, (uint64_t)ne_max * MID, sizeof(__half), "h") &&
                    vqp_grow((void **)&L->wgu, &L->wgu_cap, (uint64_t)2u * MID * IN, sizeof(__half), "wgu") &&
                    vqp_grow((void **)&L->wd, &L->wd_cap, (uint64_t)OUT * MID, sizeof(__half), "wd");
        }
        if (!grown) break;
        if (cudaMemcpy(g_vqp.perm, perm_h, (size_t)nvalid * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_vqp.inv, inv_h, (size_t)npair * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError(); break;
        }
        vqp_gather_kernel<<<nvalid, 256, 0, g_cur_stream>>>(g_vqp.xs, (const float *)x->ptr, g_vqp.perm, n_expert, IN);
        if (!cuda_ok(cudaGetLastError(), "vq prefill gather launch")) break;
        if (cudaEventRecord(g_vqp_ev_start, g_cur_stream) != cudaSuccess) { (void)cudaGetLastError(); break; }
        int bad = 0;
        for (uint32_t s = 0; s < VQP_NLANE; s++)
            if (cudaStreamWaitEvent(g_vqp_lane[s].st, g_vqp_ev_start, 0) != cudaSuccess) { (void)cudaGetLastError(); bad = 1; }
        if (bad) break;

        uint32_t li = 0;
        for (uint32_t e = 0; e < n_total_expert && !bad; e++) {
            const uint32_t ne = cnt[e];
            if (!ne) continue;
            vqp_lane *L = &g_vqp_lane[li++ % VQP_NLANE];
            const vqp_slot_hdr *h1 = &tab[(size_t)e * 3u], *h3 = h1 + 1, *h2 = h1 + 2;
            if (!h1->off || !h3->off) {
                fprintf(stderr, "ds4: [vq-prefill] L%u e=%u w1/w3 槽缺失 -- aborting\n", layer_index, e);
                bad = 1; break;
            }
            vq_dequant_kernel<<<MID, 256, 0, L->st>>>(L->wgu, blob + h1->off, MID, IN, h1->dim, h1->nc, h1->nbit, NULL);
            vq_dequant_kernel<<<MID, 256, 0, L->st>>>(L->wgu + (uint64_t)MID * IN, blob + h3->off,
                                                      MID, IN, h3->dim, h3->nc, h3->nbit, NULL);
            if (h2->off) {
                /* down 的行 = 输出通道 ⇒ 权重侧反修的增益覆盖挂在这里(只有 down 解了增益) */
                const float *gov = g_v41_gr[layer_index < 64u ? layer_index : 0];
                vq_dequant_kernel<<<OUT, 256, 0, L->st>>>(L->wd, blob + h2->off, OUT, MID, h2->dim, h2->nc, h2->nbit,
                                                          gov ? gov + (size_t)e * OUT : NULL);
            } else {
                /* 冷 w2 回退: base down 是影子张量时硬失败, 不读垃圾当权重(同 decode 路)。 */
                if (down_expert_bytes == 0 || down_offset == 0) {
                    fprintf(stderr, "ds4: [vq-prefill] L%u e=%u 冷 w2 槽缺失且 base down 不在文件里"
                                    "(影子张量) -- aborting (no silent quality downgrade)\n", layer_index, e);
                    bad = 1; break;
                }
                const uint8_t *sd = (const uint8_t *)model_map + down_offset + (uint64_t)e * down_expert_bytes;
                vq_cold_w2_kernel<<<OUT, 256, 0, L->st>>>(L->wd, sd, OUT, MID);
            }
            if (!cuda_ok(cudaGetLastError(), "vq prefill dequant launch")) { bad = 1; break; }
            const __half *xe = g_vqp.xs + (uint64_t)off[e] * IN;
            if (!vqp_gemm(L->bl, L->wgu, (int)IN, xe, (int)IN, L->gu, (int)(2u * MID),
                          (int)(2u * MID), (int)ne, (int)IN, "vq prefill gate/up gemm")) { bad = 1; break; }
            vqp_swiglu_kernel<<<ne, 256, 0, L->st>>>(L->h, L->gu, MID, clamp);
            if (!cuda_ok(cudaGetLastError(), "vq prefill swiglu launch")) { bad = 1; break; }
            if (!vqp_gemm(L->bl, L->wd, (int)MID, L->h, (int)MID, g_vqp.ys + (uint64_t)off[e] * OUT, (int)OUT,
                          (int)OUT, (int)ne, (int)MID, "vq prefill down gemm")) { bad = 1; break; }
        }
        for (uint32_t s = 0; s < VQP_NLANE; s++) {   /* 出错也要把主流拴回来, 别让下一层踩在飞的 lane */
            vqp_lane *L = &g_vqp_lane[s];
            if (cudaEventRecord(L->done, L->st) != cudaSuccess ||
                cudaStreamWaitEvent(g_cur_stream, L->done, 0) != cudaSuccess) { (void)cudaGetLastError(); bad = 1; }
        }
        if (bad) break;
        vqp_reduce_kernel<<<dim3((OUT + 255u) / 256u, n_tokens, 1), 256, 0, g_cur_stream>>>(
            (float *)out->ptr, g_vqp.ys, g_vqp.inv, (const float *)weights->ptr, n_expert, OUT);
        ok = cuda_ok(cudaGetLastError(), "vq prefill reduce launch");
        if (ok) { g_vqp_last_tok = n_tokens; g_vqp_last_used = n_expert; g_vqp_last_out = OUT; }
    } while (0);
    free(sel_h); free(perm_h); free(inv_h); free(cnt); free(off); free(cur);
    return ok;
}

/* ---- 反修取料: 逐专家 down 输出(reduce 前, 未乘路由权重) ---- */
/* out[t][k][o] = ys[inv[t·n_used+k]][o], 缺席配对填 0。与 vqp_reduce_kernel 读的是同一份 ys/inv,
 * 所以取到的就是引擎这一层真正加权求和的那些数 —— 不是另算一遍的近似。 */
__global__ static void vqp_expand_pairs_kernel(float *dst, const float *ys, const int32_t *inv,
                                               uint32_t n_used, uint32_t OUT) {
    const uint32_t pk = blockIdx.y;                 /* = t·n_used + k */
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    const int32_t i = inv[pk];
    dst[(uint64_t)pk * OUT + o] = i < 0 ? 0.0f : ys[(uint64_t)i * OUT + o];
}

int ds4_gpu_v41_vq_capture_expert_out(float *host, uint32_t n_tok, uint32_t n_used, uint32_t out_dim) {
    /* 形状必须与刚跑完的那一层逐项对上 —— 对不上说明取的不是这一层(或走的是解码 gemv 路,
     * 那条路不物化 ys)。宁可返回 0 让上层硬失败, 也不给一块"能用但对不上号"的数。 */
    if (!host || n_tok != g_vqp_last_tok || n_used != g_vqp_last_used || out_dim != g_vqp_last_out) return 0;
    const uint64_t npair = (uint64_t)n_tok * n_used, nel = npair * out_dim;
    if (!vqp_grow((void **)&g_vqp_cap, &g_vqp_cap_n, nel, sizeof(float), "vq 取料展开")) return 0;
    float *dev = g_vqp_cap;
    vqp_expand_pairs_kernel<<<dim3((out_dim + 255u) / 256u, (unsigned)npair), 256, 0, g_cur_stream>>>(
        dev, g_vqp.ys, g_vqp.inv, n_used, out_dim);
    if (!cuda_ok(cudaGetLastError(), "vq 取料展开")) return 0;
    if (cudaStreamSynchronize(g_cur_stream) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (cudaMemcpy(host, dev, (size_t)nel * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    return 1;
}

/* 装/卸某层的增益覆盖表。host=NULL 卸掉。n_expert×out_dim 必须与该层实际形状一致(调用方核过)。 */
int ds4_gpu_v41_set_gr_override(uint32_t layer, const float *host, uint32_t n_expert, uint32_t out_dim) {
    if (layer >= 64u) return 0;
    if (g_v41_gr[layer]) { (void)cudaFree(g_v41_gr[layer]); g_v41_gr[layer] = NULL; }
    if (!host) return 1;
    const size_t nb = (size_t)n_expert * out_dim * 4;
    if (cudaMalloc((void **)&g_v41_gr[layer], nb) != cudaSuccess) { (void)cudaGetLastError(); g_v41_gr[layer] = NULL; return 0; }
    if (cudaMemcpy(g_v41_gr[layer], host, nb, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError(); (void)cudaFree(g_v41_gr[layer]); g_v41_gr[layer] = NULL; return 0;
    }
    return 1;
}
