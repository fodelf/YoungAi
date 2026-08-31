/* cuda_zchain_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * go-onebit DQZ2 zchain + 路由闭式 RTE 侧车。
 */
#include <cuda_fp8.h>   /* 本分片 zc_fp8_ld 的 __nv_cvt_* 所需; 原在 ds4_cuda.cu 中段, 拆分时移到分片文首 */
/* ===== go-onebit DQZ2 zchain (CUDA) =====
 * Metal kernel_dsv4_zchain_{ge,scale} 的逐语义平移(数学契约: ds4_zchain.h)。
 * 常驻小表一次上传; dispatch 纯 kernel launch, token-graph capture 兼容。 */
static float    *g_zc_ops = NULL;        /* dev [n_ops_total][16] */
static __half   *g_zc_v8  = NULL;        /* dev concat fp16 [n_blk][8][d_model] */
static float    *g_zc_ge  = NULL;        /* dev [n_layer][n_expert] */
static uint32_t *g_zc_layer_off = NULL;  /* host [n_layer+1] */
static uint8_t  *g_zc_ge_present = NULL; /* host [n_layer] */
static uint32_t  g_zc_n_layer = 0, g_zc_n_expert = 0, g_zc_d_model = 0;
static __half   *g_zc_zlm = NULL;        /* dev packed z[k]|U[d*k]|V[din*k] per layer */
static uint32_t *g_zc_zl_off = NULL, *g_zc_zl_k = NULL, *g_zc_zl_din = NULL; /* host */
static float    *g_zc_zl_tr = NULL;      /* host */

int ds4_gpu_zchain_set(
        const float *ops, const uint32_t *layer_off, const uint16_t *v8,
        const float *ge, const uint8_t *ge_present,
        uint32_t n_layer, uint32_t n_expert, uint32_t d_model,
        uint32_t n_ops_total, uint32_t n_v8_blocks) {
    g_zc_n_layer = n_layer; g_zc_n_expert = n_expert; g_zc_d_model = d_model;
    if (n_ops_total) {
        if (!cuda_ok(cudaMalloc(&g_zc_ops, (size_t)n_ops_total * 16 * sizeof(float)), "zchain ops")) return 0;
        if (!cuda_ok(cudaMemcpy(g_zc_ops, ops, (size_t)n_ops_total * 16 * sizeof(float), cudaMemcpyHostToDevice), "zchain ops up")) return 0;
    }
    if (n_v8_blocks) {
        size_t hb = (size_t)n_v8_blocks * 8 * d_model * sizeof(__half);
        if (!cuda_ok(cudaMalloc(&g_zc_v8, hb), "zchain v8")) return 0;
        if (!cuda_ok(cudaMemcpy(g_zc_v8, v8, hb, cudaMemcpyHostToDevice), "zchain v8 up")) return 0;
    }
    if (ge && ge_present) {
        size_t gb = (size_t)n_layer * n_expert * sizeof(float);
        if (!cuda_ok(cudaMalloc(&g_zc_ge, gb), "zchain ge")) return 0;
        if (!cuda_ok(cudaMemcpy(g_zc_ge, ge, gb, cudaMemcpyHostToDevice), "zchain ge up")) return 0;
        g_zc_ge_present = (uint8_t *)malloc(n_layer);
        memcpy(g_zc_ge_present, ge_present, n_layer);
    }
    g_zc_layer_off = (uint32_t *)malloc((n_layer + 1) * sizeof(uint32_t));
    memcpy(g_zc_layer_off, layer_off, (n_layer + 1) * sizeof(uint32_t));
    fprintf(stderr, "ds4: zchain CUDA armed: %u ops, %u v8 blocks, GE=%s\n",
            n_ops_total, n_v8_blocks, g_zc_ge ? "yes" : "no");
    return 1;
}

static __global__ void zchain_ge_kernel(
        float *weights, const int *selected, const float *ge,
        uint32_t n_expert, uint32_t total, uint32_t ge_base) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int e = selected[gid];
    if (e < 0 || (uint32_t)e >= n_expert) return;
    weights[gid] *= ge[ge_base + (uint32_t)e];
}

int ds4_gpu_zchain_ge_apply(
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint32_t layer, uint32_t n_expert_used, uint32_t n_tokens) {
    if (!g_zc_ge || layer >= g_zc_n_layer || !g_zc_ge_present[layer]) return 1;
    uint32_t total = n_tokens * n_expert_used;
    zchain_ge_kernel<<<(total + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (float *)weights->ptr, (const int *)selected->ptr, g_zc_ge,
        g_zc_n_expert, total, layer * g_zc_n_expert);
    return 1;
}

#define ZC_NTG 256u
/* 一 block 每 token。shared: red[256] 树归约 | pv[<=1024] z⊙(V^T x) | ua[d<=2048]
 * U pv 投影缓存(免二遍 U 读; 与 host ds4_zchain_zl_apply 数学等价)。 */
static __device__ __forceinline__ float zc_red_add(float v, float *red) {
    uint32_t tid = threadIdx.x;
    red[tid] = v; __syncthreads();
    for (uint32_t s = ZC_NTG >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float r = red[0]; __syncthreads();
    return r;
}

static __global__ void zchain_scale_kernel(
        float *routed, const float *x, const float *ops, const __half *v8,
        const __half *zlm, uint32_t d, uint32_t n_tokens,
        uint32_t op_start, uint32_t op_count,
        uint32_t zl_k, uint32_t zl_off, uint32_t zl_din, float zl_tr, uint32_t zl_mul) {
    extern __shared__ float sh[];               /* red[ZC_NTG] | pv[zl_k] | ua[d] */
    float *red = sh, *pv = sh + ZC_NTG, *ua = pv + zl_k;
    uint32_t tok = blockIdx.x, tid = threadIdx.x;
    if (tok >= n_tokens) return;
    const float *xt = x + (uint64_t)tok * d;
    float *rt = routed + (uint64_t)tok * d;

    float acc = 0.0f;
    for (uint32_t j = tid; j < d; j += ZC_NTG) acc += xt[j] * xt[j];
    const float ss = zc_red_add(acc, red);
    const float xnorm = sqrtf(ss);

    float lam = 1.0f;
    for (uint32_t oi = 0; oi < op_count; oi++) {
        const float *op = ops + (uint64_t)(op_start + oi) * 16u;
        uint32_t ty = (uint32_t)op[0];
        if (ty == 1u) lam = op[1] * lam;
        else if (ty == 2u) {
            float c = op[2] + op[3] * ((xnorm - op[4]) / op[5]);
            c = fminf(fmaxf(c, (float)DS4_AMP_LAM_MIN), (float)DS4_AMP_LAM_MAX);
            lam = c * lam;
        } else if (ty == 3u) {
            int blk = (int)op[15];
            if (blk >= 0) {
                float c = op[6];
                for (uint32_t k = 0; k < 8u; k++) {
                    const __half *vr = v8 + ((uint64_t)blk * 8u + k) * d;
                    float dk = 0.0f;
                    for (uint32_t j = tid; j < d; j += ZC_NTG) dk += xt[j] * __half2float(vr[j]);
                    c += op[7 + k] * zc_red_add(dk, red);
                }
                c = fminf(fmaxf(c, (float)DS4_AMP_LAM_MIN), (float)DS4_AMP_LAM_MAX);
                lam = c * lam;
            }
        } else if (ty == 4u) lam = 1.0f + op[1] * (lam - 1.0f);
    }
    if (op_count) for (uint32_t j = tid; j < d; j += ZC_NTG) rt[j] *= lam;

    if (zl_k > 0u) {   /* frozen z^L: routed += clip * U diag(z) V^T x (λ 之后) */
        const uint32_t din = zl_din > 0u ? zl_din : d;
        const __half *hz = zlm + zl_off;
        /* type9(动态 z, zl_mul==3): 载荷是 A|U|V, 无 z[k] 前缀 ⇒ 偏移与 type6/7 不同 */
        const __half *hA = (zl_mul == 3u) ? (zlm + zl_off) : NULL;
        const __half *hU = (zl_mul == 3u) ? (zlm + zl_off + (uint64_t)din * zl_k) : (hz + zl_k);
        const __half *hV = hU + (uint64_t)d * zl_k;
        float nrm = 0.0f;
        if (din == 3u * d) {   /* md86 ftA: φ=[x, x⊙x/rms, relu(x)] */
            nrm = sqrtf(ss / (float)d) + 1e-6f;
        }
        for (uint32_t c = tid; c < zl_k; c += ZC_NTG) {
            float dk = 0.0f;
            if (din == 3u * d) {
                for (uint32_t j = 0; j < d; j++) {
                    float xv = xt[j];
                    dk += xv * __half2float(hV[(uint64_t)j * zl_k + c]);
                    dk += (xv * xv / nrm) * __half2float(hV[(uint64_t)(d + j) * zl_k + c]);
                    dk += (xv > 0.0f ? xv : 0.0f) * __half2float(hV[(uint64_t)(2u * d + j) * zl_k + c]);
                }
            } else {
                for (uint32_t j = 0; j < d; j++) dk += xt[j] * __half2float(hV[(uint64_t)j * zl_k + c]);
            }
            const float zs = (zl_tr > 0.0f ? zl_tr : 1.0f);
            if (zl_mul == 3u) {          /* AMPD: pv = tanh(Vᵀx/s) · tanh(Aᵀx/s), z 是 x 的函数 */
                float gk = 0.0f;
                if (din == 3u * d) {
                    for (uint32_t j = 0; j < d; j++) {
                        float xv = xt[j];
                        gk += xv * __half2float(hA[(uint64_t)j * zl_k + c]);
                        gk += (xv * xv / nrm) * __half2float(hA[(uint64_t)(d + j) * zl_k + c]);
                        gk += (xv > 0.0f ? xv : 0.0f) * __half2float(hA[(uint64_t)(2u * d + j) * zl_k + c]);
                    }
                } else {
                    for (uint32_t j = 0; j < d; j++) gk += xt[j] * __half2float(hA[(uint64_t)j * zl_k + c]);
                }
                pv[c] = tanhf(dk / zs) * tanhf(gk / zs);
            } else {
                if (zl_mul) dk = tanhf(dk / zs);   /* AMP(type7) */
                pv[c] = dk * __half2float(hz[c]);
            }
        }
        __syncthreads();
        float nd_p = 0.0f, nr_p = 0.0f;
        for (uint32_t j = tid; j < d; j += ZC_NTG) {
            const __half *ur = hU + (uint64_t)j * zl_k;
            float a = 0.0f;
            for (uint32_t c = 0; c < zl_k; c++) a += pv[c] * __half2float(ur[c]);
            ua[j] = a;
            nd_p += a * a;
            nr_p += rt[j] * rt[j];
        }
        if (zl_mul) {   /* 乘性出口(type7 AMP / type9 AMPD): ⊙(1+ua), 无信任域 */
            for (uint32_t j = tid; j < d; j += ZC_NTG) rt[j] *= (1.0f + ua[j]);
        } else {
            const float nd = sqrtf(zc_red_add(nd_p, red));
            const float nr = sqrtf(zc_red_add(nr_p, red));
            const float cap = zl_tr * nr;
            const float s = (nd > cap && nd > 0.0f) ? (cap / nd) : 1.0f;
            for (uint32_t j = tid; j < d; j += ZC_NTG) rt[j] += s * ua[j];
        }
    }
}

/* ★z^L decode 快路(2026-08-19): 老 zchain_scale_kernel 每 token 单 block, decode(n=1) 时
 * 全 GPU 只有 1 个 SM 干活且 U 段跨线程步长 k 非合并读 → 实测 ~2.7ms/层, 21.4→6.3 t/s。
 * 快路=三段网格化: pv=diag(z)Vᵀx(64列/块合并读) → ua=U·pv(warp/行合并读+范数原子归约)
 * → 信任域缩放加回。数值=同式(浮点归约序容差); 批量 prefill(n>16, 本身有 token 并行度)
 * 与 λ ops 仍走老 kernel。scratch 在 zl_set 预分配(graph capture 内禁 cudaMalloc)。 */
static float *g_zc_zl_pv = NULL, *g_zc_zl_ua = NULL, *g_zc_zl_n2 = NULL;
static float *g_zc_zl_pvp = NULL;      /* pv 两相归约 partial: MAXTOK×SEG×kmax */
static uint8_t *g_zc_zlm8 = NULL;      /* U/V fp8(e4m3) 影子(2026-08-20 ②刀): 半字节读, 省168MB/tok */
static uint32_t *g_zc_zl_mul = NULL;   /* per-layer 1=乘性AMP(type7) | 3=动态z(type9 AMPD) */
#define ZC_ZL_FAST_MAXTOK 16u
/* ★pv 占用率手术(2026-08-20)★: 老 zc_zl_pv_kernel 在 decode(n=1) 只开 k/64≈8 个 block,
 * GB10 绝大部分 SM 闲置 → nsys 实测 130µs/层(4MB V 只跑出 31GB/s), 42 层 AMP 链税 5.4ms。
 * 两相归约: A) 按 d 切 SEG 段并行出 partial(k/64×SEG×tok 个 block, 占用率拉满)
 *          B) 每列跨段求和 + tanh 定标 ×z。数值=同式(浮点归约序容差, 与老 kernel 同级)。
 * fta(φ=3d 特征抬升) 层继续走老 kernel(AMP/加性主链 din==d)。 */
#define ZC_ZL_SEG 32u   /* 08-20 提速: 16→32, pv_part 块数 128→256 再拉占用率 */

static __global__ void zc_zl_pv_kernel(
        const float *x, const __half *zlm, float *pv,
        uint32_t d, uint32_t k, uint32_t off, uint32_t fta, uint32_t mul, float mscale) {
    const uint32_t tok = blockIdx.y;
    const uint32_t tc = threadIdx.x & 63u, rg = threadIdx.x >> 6;   /* 64列×4行组 */
    const uint32_t c = blockIdx.x * 64u + tc;
    const __half *hz = zlm + off;
    const __half *hV = hz + k + (uint64_t)d * k;
    const float *xt = x + (uint64_t)tok * d;
    __shared__ float sh[4][64];
    __shared__ float r2[256];
    __shared__ float snrm;
    float nrm = 0.0f;
    if (fta) {   /* ftA 需要 rms(x): 块内自算(与 CPU zl_phi 同式) */
        float a2 = 0.0f;
        for (uint32_t j = threadIdx.x; j < d; j += 256u) a2 += xt[j] * xt[j];
        r2[threadIdx.x] = a2; __syncthreads();
        for (uint32_t s = 128u; s; s >>= 1) { if (threadIdx.x < s) r2[threadIdx.x] += r2[threadIdx.x + s]; __syncthreads(); }
        if (!threadIdx.x) snrm = sqrtf(r2[0] / (float)d) + 1e-6f;
        __syncthreads();
        nrm = snrm;
    }
    float acc = 0.0f;
    if (c < k) {
        if (fta) {
            for (uint32_t j = rg; j < d; j += 4u) {
                const float xv = xt[j];
                acc += xv * __half2float(hV[(uint64_t)j * k + c]);
                acc += (xv * xv / nrm) * __half2float(hV[(uint64_t)(d + j) * k + c]);
                acc += (xv > 0.0f ? xv : 0.0f) * __half2float(hV[(uint64_t)(2u * d + j) * k + c]);
            }
        } else {
            for (uint32_t j = rg; j < d; j += 4u)
                acc += xt[j] * __half2float(hV[(uint64_t)j * k + c]);
        }
    }
    sh[rg][tc] = acc; __syncthreads();
    if (rg == 0 && c < k) {
        float a = sh[0][tc] + sh[1][tc] + sh[2][tc] + sh[3][tc];
        if (mul) a = tanhf(a / (mscale > 0.0f ? mscale : 1.0f));   /* AMP: tanh 定标 */
        pv[(uint64_t)tok * k + c] = a * __half2float(hz[c]);
    }
}

__device__ __forceinline__ static float zc_fp8_ld(const uint8_t *p) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(*p, __NV_E4M3);
    return __half2float(*(const __half *)&hr);
}
static __global__ void zc_h2fp8_kernel(uint8_t *dst, const __half *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __nv_cvt_float_to_fp8(__half2float(src[i]), __NV_SATFINITE, __NV_E4M3);
}

static __global__ void zc_zl_pv_part8_kernel(   /* fp8 版两相 A(V 读 e4m3) */
        const float *x, const uint8_t *zlm8, float *pvp,
        uint32_t d, uint32_t k, uint32_t off) {
    const uint32_t tok = blockIdx.z, seg = blockIdx.y;
    const uint32_t tc = threadIdx.x & 63u, rg = threadIdx.x >> 6;
    const uint32_t c = blockIdx.x * 64u + tc;
    const uint8_t *hV = zlm8 + off + k + (uint64_t)d * k;
    const float *xt = x + (uint64_t)tok * d;
    const uint32_t rows = d / ZC_ZL_SEG, j0 = seg * rows, j1 = j0 + rows;
    float acc = 0.0f;
    if (c < k)
        for (uint32_t j = j0 + rg; j < j1; j += 4u)
            acc += xt[j] * zc_fp8_ld(hV + (uint64_t)j * k + c);
    __shared__ float sh[4][64];
    sh[rg][tc] = acc; __syncthreads();
    if (rg == 0 && c < k)
        pvp[((uint64_t)tok * ZC_ZL_SEG + seg) * k + c] = sh[0][tc] + sh[1][tc] + sh[2][tc] + sh[3][tc];
}

static __global__ void zc_zl_ua8_kernel(   /* fp8 版 UA(U 读 e4m3) */
        const float *routed, const float *pv, const uint8_t *zlm8, float *ua, float *n2,
        uint32_t d, uint32_t k, uint32_t off, uint32_t mul) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t j = blockIdx.x * 8u + warp;
    const uint8_t *hU = zlm8 + off + k;
    extern __shared__ float spv8[];
    for (uint32_t c = threadIdx.x; c < k; c += 256u) spv8[c] = pv[(uint64_t)tok * k + c];
    __syncthreads();
    float a = 0.0f;
    if (j < d) {
        const uint8_t *ur = hU + (uint64_t)j * k;
        for (uint32_t c = lane; c < k; c += 32u) a += spv8[c] * zc_fp8_ld(ur + c);
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    }
    if (mul) {
        if (lane == 0 && j < d) ua[(uint64_t)tok * d + j] = a;
        return;
    }
    __shared__ float snd[8], snr[8];
    if (lane == 0) {
        if (j < d) {
            ua[(uint64_t)tok * d + j] = a;
            const float r = routed[(uint64_t)tok * d + j];
            snd[warp] = a * a; snr[warp] = r * r;
        } else { snd[warp] = 0.0f; snr[warp] = 0.0f; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float nd = 0.0f, nr = 0.0f;
        for (int w = 0; w < 8; w++) { nd += snd[w]; nr += snr[w]; }
        atomicAdd(&n2[(uint64_t)tok * 2u], nd);
        atomicAdd(&n2[(uint64_t)tok * 2u + 1u], nr);
    }
}

static __global__ void zc_zl_pv_part_kernel(   /* 两相 A: partial[seg][c]=Σ_{j∈seg} x_j·V[j,c]
     * v3(2026-08-20 顺序流重排): 旧版每 j 只读 128B 且 rg 跳 4 行(4KB 跨步)实测 90GB/s;
     * 现一 block 承包段内全行×全列: 行内 k×2B 连续+行间顺序=纯流。线程持列 {t,t+256,..}。 */
        const float *x, const __half *zlm, float *pvp,
        uint32_t d, uint32_t k, uint32_t off) {
    const uint32_t tok = blockIdx.z, seg = blockIdx.y;
    const __half *hV = zlm + off + k + (uint64_t)d * k;
    const float *xt = x + (uint64_t)tok * d;
    const uint32_t rows = d / ZC_ZL_SEG, j0 = seg * rows;
    /* v5: 64列对×4行组/块 + half2 满线事务(v2 病=单half 64B半线; v3/v4 病=32块喂不饱
     * GB10 延迟). grid=(k2/64, SEG, ntok)≈128 块。 */
    const uint32_t tc = threadIdx.x & 63u, rg = threadIdx.x >> 6;
    const uint32_t kh = k >> 1;
    const uint32_t cp = blockIdx.x * 64u + tc;      /* half2 列对号 */
    float2 acc = {0.0f, 0.0f};
    if (cp < kh) {
        const uint32_t j1 = j0 + rows;
        for (uint32_t j = j0 + rg; j < j1; j += 4u) {
            const __half2 v = ((const __half2 *)(hV + (uint64_t)j * k))[cp];
            const float xv = xt[j];
            acc.x += xv * __low2float(v);
            acc.y += xv * __high2float(v);
        }
    }
    __shared__ float2 sh2[4][64];
    sh2[rg][tc] = acc; __syncthreads();
    if (rg == 0 && cp < kh) {
        const float2 a0 = sh2[0][tc], a1 = sh2[1][tc], a2 = sh2[2][tc], a3 = sh2[3][tc];
        const uint64_t base = ((uint64_t)tok * ZC_ZL_SEG + seg) * k + 2u * cp;
        pvp[base] = a0.x + a1.x + a2.x + a3.x;
        pvp[base + 1u] = a0.y + a1.y + a2.y + a3.y;
    }
}

static __global__ void zc_zl_pv_reduce_kernel(   /* 两相 B: 跨段求和 + AMP tanh 定标 ×z */
        const __half *zlm, const float *pvp, float *pv,
        uint32_t k, uint32_t off, uint32_t mul, float mscale) {
    const uint32_t tok = blockIdx.y;
    const uint32_t c = blockIdx.x * 256u + threadIdx.x;
    if (c >= k) return;
    float a = 0.0f;
    for (uint32_t s = 0; s < ZC_ZL_SEG; s++)
        a += pvp[((uint64_t)tok * ZC_ZL_SEG + s) * k + c];
    if (mul) a = tanhf(a / (mscale > 0.0f ? mscale : 1.0f));
    pv[(uint64_t)tok * k + c] = a * __half2float(zlm[off + c]);
}

static __global__ void zc_zl_ua_kernel(
        const float *routed, const float *pv, const __half *zlm, float *ua, float *n2,
        uint32_t d, uint32_t k, uint32_t off, uint32_t mul) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t j = blockIdx.x * 8u + warp;   /* warp/行: ur[c..] 合并读 */
    const __half *hU = zlm + off + k;
    extern __shared__ float spv[];   /* pv 预载共享内存: 免 8 行重复读全局 */
    for (uint32_t c = threadIdx.x; c < k; c += 256u) spv[c] = pv[(uint64_t)tok * k + c];
    __syncthreads();
    float a = 0.0f;
    if (j < d) {
        const __half *ur = hU + (uint64_t)j * k;
        for (uint32_t c = lane; c < k; c += 32u) a += spv[c] * __half2float(ur[c]);
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    }
    if (mul) {   /* AMP 乘性: 无信任域 → 免范数归约/原子 */
        if (lane == 0 && j < d) ua[(uint64_t)tok * d + j] = a;
        return;
    }
    __shared__ float snd[8], snr[8];
    if (lane == 0) {
        if (j < d) {
            ua[(uint64_t)tok * d + j] = a;
            const float r = routed[(uint64_t)tok * d + j];
            snd[warp] = a * a; snr[warp] = r * r;
        } else { snd[warp] = 0.0f; snr[warp] = 0.0f; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float nd = 0.0f, nr = 0.0f;
        for (int w = 0; w < 8; w++) { nd += snd[w]; nr += snr[w]; }
        atomicAdd(&n2[(uint64_t)tok * 2u], nd);
        atomicAdd(&n2[(uint64_t)tok * 2u + 1u], nr);
    }
}

static __global__ void zc_zl_add_kernel(
        float *routed, const float *ua, const float *n2, uint32_t d, float tr, uint32_t mul) {
    const uint32_t tok = blockIdx.y;
    const uint32_t j = blockIdx.x * 256u + threadIdx.x;
    if (j >= d) return;
    if (mul) {   /* AMP 乘性出口: ⊙(1+ua), 无信任域 */
        routed[(uint64_t)tok * d + j] *= (1.0f + ua[(uint64_t)tok * d + j]);
        return;
    }
    const float nd = sqrtf(n2[(uint64_t)tok * 2u]);
    const float nr = sqrtf(n2[(uint64_t)tok * 2u + 1u]);
    const float cap = tr * nr;
    const float s = (nd > cap && nd > 0.0f) ? cap / nd : 1.0f;
    routed[(uint64_t)tok * d + j] += s * ua[(uint64_t)tok * d + j];
}

int ds4_gpu_zchain_scale_routed(
        ds4_gpu_tensor *routed, const ds4_gpu_tensor *x,
        uint32_t layer, uint32_t n_tokens) {
    if (layer >= g_zc_n_layer || !g_zc_layer_off) return 1;
    uint32_t op_start = g_zc_layer_off[layer];
    uint32_t op_count = g_zc_layer_off[layer + 1] - op_start;
    uint32_t zk  = g_zc_zl_k   ? g_zc_zl_k[layer]   : 0;
    uint32_t zo  = g_zc_zl_off ? g_zc_zl_off[layer] : 0;
    uint32_t zd  = g_zc_zl_din ? g_zc_zl_din[layer] : 0;
    float    ztr = g_zc_zl_tr  ? g_zc_zl_tr[layer]  : 0.0f;
    if (!op_count && !zk) return 1;
    /* decode 快路的 fp8 影子按 z|U|V 布局打包, type9(AMPD) 是 A|U|V —— 布局不同, 先走通用核
     * (数值同式, 只是 decode 慢一档)。快路的 AMPD 版待做。 */
    const uint32_t zmul_l = g_zc_zl_mul ? g_zc_zl_mul[layer] : 0u;
    if (zk && zmul_l != 3u && n_tokens <= ZC_ZL_FAST_MAXTOK && g_zc_zl_pv &&
        g_zc_d_model <= 4096u && zk <= 1024u) {
        if (op_count) {   /* λ ops 仍走老 kernel(zk=0 抑制其 zl 段) */
            size_t shmem0 = (ZC_NTG + g_zc_d_model) * sizeof(float);
            zchain_scale_kernel<<<n_tokens, ZC_NTG, shmem0, g_cur_stream>>>(
                (float *)routed->ptr, (const float *)x->ptr, g_zc_ops, g_zc_v8, g_zc_zlm,
                g_zc_d_model, n_tokens, op_start, op_count, 0u, 0u, 0u, 0.0f, 0u);
        }
        const uint32_t d = g_zc_d_model;
        const uint32_t fta = (zd == 3u * d) ? 1u : 0u;
        const uint32_t zmul = zmul_l;
        if (!zmul)   /* AMP 无信任域不消费 n2 → 免每层 memset */
            cudaMemsetAsync(g_zc_zl_n2, 0, (size_t)n_tokens * 2u * sizeof(float), g_cur_stream);
        const uint32_t use8 = (g_zc_zlm8 != NULL) ? 1u : 0u;   /* fp8 影子在 = 走半字节路 */
        const uint32_t v4ok = (((zo + zk) & 1u) == 0u && (zk & 1u) == 0u);
        if (!fta && (d % ZC_ZL_SEG) == 0u && g_zc_zl_pvp && (use8 || v4ok)) {   /* 两相归约 */
            if (use8) {
                dim3 ga((zk + 63u) / 64u, ZC_ZL_SEG, n_tokens);
                zc_zl_pv_part8_kernel<<<ga, 256u, 0, g_cur_stream>>>(
                    (const float *)x->ptr, g_zc_zlm8, g_zc_zl_pvp, d, zk, zo);
            } else {
                /* v5: 64列对×4行组; half2 需 V 基址 4B 对齐(off+k 偶, v4ok 已闸) */
                dim3 ga((zk / 2u + 63u) / 64u, ZC_ZL_SEG, n_tokens);
                zc_zl_pv_part_kernel<<<ga, 256u, 0, g_cur_stream>>>(
                    (const float *)x->ptr, g_zc_zlm, g_zc_zl_pvp, d, zk, zo);
            }
            dim3 gb((zk + 255u) / 256u, n_tokens);
            zc_zl_pv_reduce_kernel<<<gb, 256u, 0, g_cur_stream>>>(
                g_zc_zlm, g_zc_zl_pvp, g_zc_zl_pv, zk, zo, zmul, ztr);
        } else {
            dim3 g1((zk + 63u) / 64u, n_tokens);
            zc_zl_pv_kernel<<<g1, 256u, 0, g_cur_stream>>>(
                (const float *)x->ptr, g_zc_zlm, g_zc_zl_pv, d, zk, zo, fta, zmul, ztr);
        }
        dim3 g2((d + 7u) / 8u, n_tokens);
        if (use8 && !fta)
            zc_zl_ua8_kernel<<<g2, 256u, (size_t)zk * sizeof(float), g_cur_stream>>>(
                (const float *)routed->ptr, g_zc_zl_pv, g_zc_zlm8, g_zc_zl_ua, g_zc_zl_n2, d, zk, zo, zmul);
        else
            zc_zl_ua_kernel<<<g2, 256u, (size_t)zk * sizeof(float), g_cur_stream>>>(
                (const float *)routed->ptr, g_zc_zl_pv, g_zc_zlm, g_zc_zl_ua, g_zc_zl_n2, d, zk, zo, zmul);
        dim3 g3((d + 255u) / 256u, n_tokens);
        zc_zl_add_kernel<<<g3, 256u, 0, g_cur_stream>>>(
            (float *)routed->ptr, g_zc_zl_ua, g_zc_zl_n2, d, ztr, zmul);
        return 1;
    }
    size_t shmem = (ZC_NTG + zk + g_zc_d_model) * sizeof(float);
    zchain_scale_kernel<<<n_tokens, ZC_NTG, shmem, g_cur_stream>>>(
        (float *)routed->ptr, (const float *)x->ptr, g_zc_ops, g_zc_v8, g_zc_zlm,
        g_zc_d_model, n_tokens, op_start, op_count, zk, zo, zd, ztr,
        g_zc_zl_mul ? g_zc_zl_mul[layer] : 0u);
    return 1;
}

