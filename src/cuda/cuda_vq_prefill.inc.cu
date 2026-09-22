/* cuda_vq_prefill.inc.cu — VQ 专家批量(prefill)路的【调度骨架】(2026-09-06 建, 09-15 清到只剩骨架)。
 *
 * 这个文件不做乘法。它干的是乘法之前和之后的事:
 *   ① (token,pick) 对按专家做稳定计数排序 —— 同一个专家名下的激活行排到一起, 一个专家的权重只碰一次;
 *   ② 每层一次把 blob 里 384×3 个槽的头抄到主机缓存(vqp_hdr_build), 免得主机逐个碰 mmap 缺页;
 *   ③ 算完之后按 token 以固定 pick 序加权求和(vqp_reduce_kernel, 定序 ⇒ 同输入逐位可复现);
 *   ④ 反修取料(逐专家 down 输出)与 gr 覆盖。
 * 真正的乘法在 vqp_fused_run(cuda_vq_prefill_fused.inc.cu → cuda_vq_prefill_nvfp4.inc.cu):
 * VQ 位流直接解成 NVFP4 喂板子的 FP4 张量核, 不落任何 f16 暂存。
 *
 * ★2026-09-15 clear.md C0 删掉的: 老的"逐专家 dequant 成 f16 → cuBLAS f16 GEMM"那条路★
 * (vqp_gemm / vqp_gather_kernel(f32→f16) / vqp_swiglu_kernel(出 f16) / g_vqp.xs 暂存 / 4 条侧流 lane)。
 * 09-15 第三轮起预填专家已走融合路, 这些件再没被调用过, 但侧流初始化还在每次跑时建 4 个流 +
 * 4 个 cuBLAS 句柄 + 5 个事件。删它不改任何数值。
 *
 * 改了会怎样: reduce 若改成 atomicAdd 会回到 08-22 删掉的"同 prompt 两次跑不同"; 排序若丢了稳定性
 * (同专家内不保持 token 序)同样不可复现 —— 两者都不报错, 只是同一份输入跑出两个 PPL。 */

/* 权重侧反修(--zchain 目录里的 gr_Lnn.bin): [layer] → 设备上的 s[n_expert][OUT] 缩放因子, NULL = 该层不挂。
 * 只作用在 down 的行增益上(行 = 输出通道)。挂了就 100% 生效, 不做任何静默回退。 */
static float *g_v41_gr[64];

/* 融合路(cuda_vq_prefill_fused.inc.cu, 同一 TU 后面定义): VQ 解码即乘, 不落 f16 暂存。
 * 2026-09-15 第三轮起它是【唯一】的预填专家路 —— 逐专家 dequant + cuBLAS 那条已删(算术强度只有 8,
 * 为 8 个 token 解一整份权重, 见新文件头的账)。 */
static int vqp_fused_run(const uint8_t *blob, const uint32_t *cnt, const uint32_t *off_h, uint32_t n_total_expert,
                         uint32_t nvalid, uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t nc, float clamp,
                         const float *x, const int32_t *perm, uint32_t n_expert, uint32_t layer_index, uint32_t ver);

static struct {
    float   *ys;   uint64_t ys_cap;    /* 排序后的 down 输出 [nvalid][OUT] f32 */
    int32_t *perm; uint64_t perm_cap;  /* 排序位置 → pair 编号 */
    int32_t *inv;  uint64_t inv_cap;   /* pair 编号 → 排序位置(-1=无效 pick) */
} g_vqp;

/* 反修取料(2026-09-13): 上一次 prefill MoE 的形状 + 展开缓冲。形状用来校验取料方要的层对不对得上。 */
static uint32_t g_vqp_last_tok = 0, g_vqp_last_used = 0, g_vqp_last_out = 0;
static float *g_vqp_cap = NULL; static uint64_t g_vqp_cap_n = 0;

/* ★2026-09-15 clear.md C0: 侧流流水(4 lane × 流/cuBLAS 句柄/事件 + 每 lane 的 f16 权重暂存)已删★
 * 它是"逐专家 dequant 成 f16 → cuBLAS"那条路的配套: 让带宽型 dequant 与算力型 GEMM 重叠。
 * 09-15 第三轮起预填专家走融合/NVFP4, 权重不再落 f16 暂存, 这 4 条 lane 就再没人往里发过东西 ——
 * 但初始化还在每次跑时建 4 个流 + 4 个 cuBLAS 句柄 + 5 个事件(cuBLAS 句柄各自带 workspace)。
 * 删它不改任何数值: 全路径都在 g_cur_stream 上。 */

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
                         uint32_t IN, uint32_t MID, uint32_t OUT, uint32_t ver) {
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
        /* 期望的载荷魔数按【盘上版本】定(2026-09-21): v3 的载荷是 'DQV3'(布局不同, 见 cuda_vq_row.inc.cu)。
         * 写死 v2 的话 v3 文件会在这里报"头错"并 abort —— 那倒是安全的失败, 但预填就整个不可用了。 */
        if (d[2] != (ver == 3u ? DS4VQ_MAT3_MAGIC : DS4VQ_MAT_MAGIC) || d[4] != exp_rows || d[5] != exp_cols) {
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

static int cuda_vq_moe_prefill_gemm(
        ds4_gpu_tensor *out, const uint8_t *blob,
        const void *model_map, uint64_t down_offset, uint64_t down_expert_bytes,
        uint32_t IN, uint32_t MID, uint32_t OUT,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, uint32_t ver) {
    if (!g_cublas_ready) { fprintf(stderr, "ds4: [vq-prefill] cuBLAS 未就绪 (L%u)\n", layer_index); return 0; }
    if (!vqp_hdr_build(layer_index, blob, n_total_expert, IN, MID, OUT, ver)) return 0;
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
        /* +V41_VQN_PAD 行: NVFP4 路把 GEMM 的 n 对齐到 2 的幂, 最后一个专家的补齐行落在这里 */
        if (!vqp_grow((void **)&g_vqp.ys, &g_vqp.ys_cap, ((uint64_t)nvalid + 2048u) * OUT, sizeof(float), "ys") ||
            !vqp_grow((void **)&g_vqp.perm, &g_vqp.perm_cap, nvalid, sizeof(int32_t), "perm") ||
            !vqp_grow((void **)&g_vqp.inv, &g_vqp.inv_cap, npair, sizeof(int32_t), "inv")) break;
        (void)ne_max;   /* 融合路按工作项(≤8 token)开寄存器累加器, 不需要按最大专家宽度预分配 */
        if (cudaMemcpy(g_vqp.perm, perm_h, (size_t)nvalid * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g_vqp.inv, inv_h, (size_t)npair * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError(); break;
        }
        /* ★2026-09-15 第三轮: 逐专家 dequant→cuBLAS 换成融合路(解码即乘)★
         * 先在主机把槽表查一遍: 有 token 的专家缺 w1/w3/w2 槽就硬失败, 不让核里写 0 悄悄降质。 */
        int bad = 0;
        for (uint32_t e = 0; e < n_total_expert && !bad; e++) {
            if (!cnt[e]) continue;
            const vqp_slot_hdr *h1 = &tab[(size_t)e * 3u];
            if (!h1->off || !(h1 + 1)->off || !(h1 + 2)->off) {
                fprintf(stderr, "ds4: [vq-prefill] L%u e=%u 槽缺失(w1/w3/w2) -- aborting (no silent quality downgrade)\n",
                        layer_index, e);
                bad = 1;
            }
        }
        if (bad) break;
        (void)model_map; (void)down_offset; (void)down_expert_bytes;
        if (!vqp_fused_run(blob, cnt, off, n_total_expert, nvalid, IN, MID, OUT,
                           tab[0].nc, clamp, (const float *)x->ptr, g_vqp.perm, n_expert, layer_index, ver)) break;
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
