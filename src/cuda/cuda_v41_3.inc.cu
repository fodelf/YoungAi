/* cuda_v41_3.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 批前向原语 ③(2026-09-12 战役 P2)。
 * 路由(Gate sqrtsoftplus + bias topk) / SwiGLU(Expert 截断语义) / routed MoE(VQ blob → cuda_vq_moe_prefill_gemm)。 */

/* 官方 Gate.forward: scores = linear(x.float(), w.float()); probs = sqrt(softplus(scores));
 * indices = topk(probs + bias); weights = probs[indices] / (Σ + 1e-20) · route_scale。
 * 一 warp 一 token: lane 持 12 个专家(384/32), topk 每轮一次 warp argmax(同分取小号, 与顺序扫同语义)。
 * 第一版一 block 一 token、线程 0 串行 6×384 扫: nsys 实测 400 µs/层 = 16 ms/token。 */
#define V41_ROUTER_PER_LANE 12u
__global__ static void v41_router_kernel(int32_t *sel, float *wts, const float *logits, const float *bias,
                                         uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31u;
    if (t >= n_tok) return;
    const float *lg = logits + (uint64_t)t * n_expert;
    float pr[V41_ROUTER_PER_LANE], sc[V41_ROUTER_PER_LANE];
    #pragma unroll
    for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) {
        const uint32_t e = lane + 32u * k;
        if (e < n_expert) {
            const float z = lg[e];
            const float sp = z > 20.0f ? z : log1pf(expf(z));   /* softplus(β=1, threshold 20 与 torch 同) */
            pr[k] = sqrtf(sp); sc[k] = pr[k] + bias[e];
        } else { pr[k] = 0.f; sc[k] = -INFINITY; }
    }
    uint32_t used = 0; float wsum = 0.f;
    for (uint32_t r = 0; r < topk; r++) {
        float bv = -INFINITY; int bk = -1;
        #pragma unroll
        for (uint32_t k = 0; k < V41_ROUTER_PER_LANE; k++) if (!((used >> k) & 1u) && sc[k] > bv) { bv = sc[k]; bk = (int)k; }
        int be = bk >= 0 ? (int)(lane + 32u * (uint32_t)bk) : -1;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off); const int oe = __shfl_xor_sync(0xffffffffu, be, off);
            if (oe >= 0 && (be < 0 || ov > bv || (ov == bv && oe < be))) { bv = ov; be = oe; }
        }
        float myp = 0.f;
        if (be >= 0 && (uint32_t)(be & 31) == lane) { const uint32_t k = (uint32_t)be >> 5; used |= 1u << k; myp = pr[k]; }
        for (int off = 16; off > 0; off >>= 1) myp += __shfl_xor_sync(0xffffffffu, myp, off);   /* 只有一个 lane 非零 */
        /* ★每轮直接落盘(2026-09-18)★: 原来先攒进 chosen[16]/chosen_p[16] 两个数组、末尾再循环写出 —— 运行期下标(topk)让
         * 它们掉进 local memory。先存未归一的 myp, 最后一遍再除 wsum: 算式 myp/(wsum+1e-20)·route_scale 一个字没动, 逐位同。 */
        if (lane == 0) { sel[(uint64_t)t * topk + r] = be; wts[(uint64_t)t * topk + r] = myp; }
        wsum += myp;
    }
    if (lane == 0)
        for (uint32_t r = 0; r < topk; r++) wts[(uint64_t)t * topk + r] = wts[(uint64_t)t * topk + r] / (wsum + 1e-20f) * route_scale;
}
/* 路由偏置侧车(2026-09-20): [exp_probs_b 文件偏移] → 设备上的 (盘上 bias + Δb)[E]。按偏移认层: 主干 40 层与三塔各自的
 * exp_probs_b 偏移互不相同, 路由入口不用改签名。挂了就 100% 生效, 不做静默回退; 盘上文件一个字节不动。 */
static struct { uint64_t off; float *dev; uint32_t n; } g_v41_rb[64];
static uint32_t g_v41_rb_n = 0;
int ds4_gpu_v41_set_rb_override(const void *model_map, uint64_t model_size, uint64_t bias_offset, const float *host_delta, uint32_t n_expert) {
    if (!host_delta) {   /* 卸: offset 0 = 全卸 */
        for (uint32_t i = 0; i < g_v41_rb_n;) {
            if (bias_offset == 0 || g_v41_rb[i].off == bias_offset) { (void)cudaFree(g_v41_rb[i].dev); g_v41_rb[i] = g_v41_rb[--g_v41_rb_n]; }
            else i++;
        }
        return 1;
    }
    if (!model_map || !n_expert || bias_offset > model_size || (uint64_t)n_expert * 4 > model_size - bias_offset) return 0;
    const float *base = (const float *)cuda_model_range_ptr(model_map, bias_offset, (uint64_t)n_expert * 4, "v41 rb base");
    if (!base) return 0;
    float *h = (float *)malloc((size_t)n_expert * 4);
    if (!h) return 0;
    /* 盘上 bias 可能在设备副本也可能在主机映射: cudaMemcpyDefault 按 UVA 认指针 */
    if (cudaMemcpy(h, base, (size_t)n_expert * 4, cudaMemcpyDefault) != cudaSuccess) { (void)cudaGetLastError(); free(h); return 0; }
    for (uint32_t e = 0; e < n_expert; e++) h[e] += host_delta[e];
    uint32_t i = 0;
    for (; i < g_v41_rb_n; i++) if (g_v41_rb[i].off == bias_offset) break;
    if (i == g_v41_rb_n) {
        if (g_v41_rb_n >= 64u) { free(h); return 0; }
        if (cudaMalloc((void **)&g_v41_rb[i].dev, (size_t)n_expert * 4) != cudaSuccess) { (void)cudaGetLastError(); free(h); return 0; }
        g_v41_rb[i].off = bias_offset; g_v41_rb[i].n = n_expert; g_v41_rb_n++;
    } else if (g_v41_rb[i].n != n_expert) { free(h); return 0; }
    const int ok = cudaMemcpy(g_v41_rb[i].dev, h, (size_t)n_expert * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) (void)cudaGetLastError();
    free(h);
    return ok;
}

int ds4_gpu_v41_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
                              const void *model_map, uint64_t model_size, uint64_t bias_offset,
                              uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    if (!selected || !weights || !logits || topk > 16u || n_expert > 32u * V41_ROUTER_PER_LANE) return 0;
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, (uint64_t)n_expert * 4, "v41 gate bias");
    if (!bias) return 0;
    for (uint32_t i = 0; i < g_v41_rb_n; i++) if (g_v41_rb[i].off == bias_offset && g_v41_rb[i].n == n_expert) { bias = g_v41_rb[i].dev; break; }
    v41_router_kernel<<<(n_tok + 7u) / 8u, 256, 0, g_cur_stream>>>((int32_t *)selected->ptr, (float *)weights->ptr,
        (const float *)logits->ptr, bias, n_tok, n_expert, topk, route_scale);
    return cuda_ok(cudaGetLastError(), "v41 router");
}

/* 官方 Expert.forward: gate=w1(x).float(); up=w3(x).float(); up=clamp(±limit); gate=clamp(max=limit);
 * h = silu(gate)·up → (×路由权重在外面) → .to(bf16)。这里 gate/up 是 fp4 线性的 bf16 输出(调用方已舍)。 */
__global__ static void v41_swiglu_kernel(float *h, const float *g, const float *u, uint64_t n, float limit) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) { uv = fminf(fmaxf(uv, -limit), limit); gv = fminf(gv, limit); }
    h[i] = v41_bf16r((gv / (1.0f + expf(-gv))) * uv);
}
int ds4_gpu_v41_swiglu_tensor(ds4_gpu_tensor *h, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                              uint32_t n_tok, uint32_t mid, float limit) {
    if (!h || !gate || !up) return 0;
    const uint64_t n = (uint64_t)n_tok * mid;
    v41_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)h->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, limit);
    return cuda_ok(cudaGetLastError(), "v41 swiglu");
}

/* engram 门(官方 Engram.forward): 一 block 一 (token, 路); 256 线程归约 h², key², h·w·key 三个和 */
__global__ static void v41_engram_gate_kernel(float *hc, const float *kv, const float *qw, const float *kw,
                                              uint32_t E, uint32_t n_hc, float eps) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t t = blockIdx.y, c = blockIdx.x;
    float *h = hc + ((uint64_t)t * n_hc + c) * E;
    const float *key = kv + (uint64_t)t * (n_hc + 1u) * E + (uint64_t)c * E;
    const float *val = kv + (uint64_t)t * (n_hc + 1u) * E + (uint64_t)n_hc * E;
    float sh = 0.f, sk = 0.f, sd = 0.f;
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) {
        const float hv = h[d], kv_ = key[d], w = qw[c * E + d] * kw[c * E + d];
        sh += hv * hv; sk += kv_ * kv_; sd += hv * w * kv_;
    }
    __shared__ float r[3][256];
    r[0][threadIdx.x] = sh; r[1][threadIdx.x] = sk; r[2][threadIdx.x] = sd; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) { r[0][threadIdx.x] += r[0][threadIdx.x + k]; r[1][threadIdx.x] += r[1][threadIdx.x + k]; r[2][threadIdx.x] += r[2][threadIdx.x + k]; }
        __syncthreads();
    }
    const float rstd = rsqrtf(r[0][0] / (float)E + eps) * rsqrtf(r[1][0] / (float)E + eps);
    const float dot = r[2][0] * rstd * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float z = copysignf(mag, dot);
    const float gate = 1.0f / (1.0f + expf(-z));
    for (uint32_t d = threadIdx.x; d < E; d += blockDim.x) h[d] = v41_bf16r(h[d] + gate * val[d]);
}
int ds4_gpu_v41_engram_gate_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size,
                                   uint64_t q_w_offset, uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    if (!hc || !kv) return 0;
    const float *qw = (const float *)cuda_model_range_ptr(model_map, q_w_offset, (uint64_t)n_hc * n_embd * 4, "v41 engram q");
    const float *kw = (const float *)cuda_model_range_ptr(model_map, k_w_offset, (uint64_t)n_hc * n_embd * 4, "v41 engram k");
    if (!qw || !kw) return 0;
    v41_engram_gate_kernel<<<dim3(n_hc, n_tok), 256, 0, g_cur_stream>>>((float *)hc->ptr, (const float *)kv->ptr, qw, kw, n_embd, n_hc, eps);
    return cuda_ok(cudaGetLastError(), "v41 engram gate");
}

/* routed MoE: 走现役 VQ prefill GEMM 路(按专家排序、逐专家 dequant f16、cuBLAS、定序 reduce)。
 * ★blob 的设备指针自己注册、用完即注销★(2026-09-12): 40 层 blob 共 98 GiB, 走 cuda_model_range_ptr 的
 * 懒注册会一层一层把整个文件钉死在内存(pinned 不可回收) —— 单机 121 GB 装不下 103 GiB 模型 + 运行时。
 * 这里每层: cudaHostRegister(该层 2.6 GB) → 算 → 同步 → 注销 → madvise(DONTNEED) 让页缓存可回收。
 * 代价是每趟前向从盘重读专家(对拍期可接受; 常驻/流式策略是 P4 的事)。
 * down 槽在 DQVL v2 里必在(which=2), 冷 w2 回退不触发。 */
static struct { uintptr_t reg; uint64_t bytes; void *dev; int valid; } g_v41_stream_reg;   /* 流式路当前钉住的那一层 */
int ds4_gpu_v41_routed_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t blob_offset, uint64_t blob_bytes,
                                  uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                                  const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                                  uint32_t n_total_expert, uint32_t n_expert_used, float clamp,
                                  const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tok) {
    /* out == NULL(只许解码小批): 不做最后那发归约, 专家部分和留在暂存, 调用方随后用 ds4_gpu_v41_moe_tail_tensor 一发收尾 */
    if ((!out && n_tok > V41_GEMV_MAX_TOK) || !selected || !weights || !x || n_tok == 0) return 0;
    if (blob_offset > model_size || blob_bytes > model_size - blob_offset) return 0;
    /* 引擎启动若已把整个 mmap 注册成设备可见(spark 实测: offload 模式下也注册了 103 GiB 整映射),
     * blob 就直接拿 range 指针 —— 再 cudaHostRegister 同一段会报 AlreadyRegistered(09-12 首跑即此)。
     * 只有整映射未注册时才走下面"自己注册→算→注销→DONTNEED"的流式路。 */
    /* 码本词数与【盘上版本】都从主机侧 blob 头读: 词数在专家 0 的 w1 载荷 +6 处(u16), 版本在 blob 头 +4(u32)。
     * ★版本必须读, 不许猜★(2026-09-21): v2 与 v3 的载荷布局不同(v3 码本一层一本、位流 12 位主流 + 位平面),
     * 选错实例不报错、只出一整套假权重。ver 不在白名单就硬停 —— 加载期的错要在加载期炸。 */
    uint32_t nc = 0, ver = 0;
    {
        const uint8_t *bh = (const uint8_t *)model_map + blob_offset;
        if (!ds4vq_blob_ok(bh, (size_t)blob_bytes)) {
            fprintf(stderr, "ds4: [v41] L%u 专家 blob 头不合法(魔数/版本/专家数); 本引擎认 DQVL v%u~v%u\n",
                    layer, DS4VQ_BLOB_VER_MIN, DS4VQ_BLOB_VER_MAX);
            exit(1);
        }
        ver = ds4vq_blob_ver(bh);
        uint64_t off0; memcpy(&off0, bh + 16, 8);
        if (off0 && off0 + 8 <= blob_bytes) { uint16_t n16; memcpy(&n16, bh + off0 + 6, 2); nc = n16; }
    }
    /* 缓存命中优先(2026-09-20): 封顶模式下整映射不注册, 装进缓存的层仍从设备副本读; 未命中而整映射已注册/已拷
     * ⇒ 旧路(UVA 映射指针, 与改前逐字节同); 两者皆无 ⇒ 下面的逐层流式路 —— 封顶模式装不下的层就靠这一条。 */
    const uint8_t *blob = (const uint8_t *)cuda_model_range_cached_ptr(model_map, blob_offset, blob_bytes);
    if (!blob && (g_model_registered || g_model_device_owned)) {
        blob = (const uint8_t *)cuda_model_range_ptr(model_map, blob_offset, blob_bytes, "v41 vq blob");
        if (!blob) return 0;
    }
    if (blob) {
        if (n_tok <= V41_GEMV_MAX_TOK)   /* 解码小批: 码本在核内查表即乘, 不落 f16(cuda_vq_decode.inc.cu) */
            return v41_vq_fused_moe(out ? (float *)out->ptr : NULL, blob, in_dim, mid_dim, out_dim, (const int32_t *)selected->ptr,
                                    (const float *)weights->ptr, n_expert_used, clamp, (const float *)x->ptr, n_tok, nc,
                                    layer < 64u ? g_v41_gr[layer] : NULL, ver);
        return cuda_vq_moe_prefill_gemm(out, blob, model_map, 0, 0, in_dim, mid_dim, out_dim, selected, weights,
                                        n_total_expert, n_expert_used, clamp, x, layer, n_tok, ver);
    }
    const long page_l = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_l > 0 ? (uint64_t)page_l : 4096u;
    const uintptr_t host = (uintptr_t)((const char *)model_map + blob_offset);
    const uintptr_t reg = host & ~(uintptr_t)(page - 1u);
    const uint64_t delta = (uint64_t)(host - reg);
    const uint64_t reg_bytes = (delta + blob_bytes + page - 1u) & ~(page - 1u);
    /* 登记只在换层时做(2026-09-20 封顶模式): 8192 token 按 512 分块 ⇒ 同一层一趟被叫 16 次, 每次注册/注销 2.6 GB
     * 光页表就是秒级; 只钉住"当前这一层", 换层才注销上一层。注销后不再 madvise(DONTNEED): 那些页只是普通页缓存,
     * 内核按压力回收, 留着 = 下一趟不必从 NVMe 重读(可用内存账里它们本来就算 available)。 */
    if (!g_v41_stream_reg.valid || g_v41_stream_reg.reg != reg || g_v41_stream_reg.bytes != reg_bytes) {
        if (g_v41_stream_reg.valid) {
            (void)cudaDeviceSynchronize();      /* 侧流 lane 可能还在读上一层 blob; 注销前必须全部落地 */
            (void)cudaHostUnregister((void *)g_v41_stream_reg.reg);
            g_v41_stream_reg.valid = 0;
        }
        void *dev = NULL;
        if (cudaHostRegister((void *)reg, (size_t)reg_bytes, cudaHostRegisterMapped) != cudaSuccess) {
            (void)cudaGetLastError();
            fprintf(stderr, "ds4: [v41] L%u blob cudaHostRegister 失败 (%.2f GB)\n", layer, (double)reg_bytes / 1e9);
            return 0;
        }
        if (cudaHostGetDevicePointer(&dev, (void *)reg, 0) != cudaSuccess || !dev) {
            (void)cudaGetLastError(); (void)cudaHostUnregister((void *)reg);
            fprintf(stderr, "ds4: [v41] L%u blob 设备指针失败\n", layer);
            return 0;
        }
        g_v41_stream_reg.reg = reg; g_v41_stream_reg.bytes = reg_bytes; g_v41_stream_reg.dev = dev; g_v41_stream_reg.valid = 1;
    }
    blob = (const uint8_t *)g_v41_stream_reg.dev + delta;
    const int ok = n_tok <= V41_GEMV_MAX_TOK
        ? v41_vq_fused_moe(out ? (float *)out->ptr : NULL, blob, in_dim, mid_dim, out_dim, (const int32_t *)selected->ptr,
                       (const float *)weights->ptr, n_expert_used, clamp, (const float *)x->ptr, n_tok, nc,
                       layer < 64u ? g_v41_gr[layer] : NULL, ver)
        : cuda_vq_moe_prefill_gemm(out, blob, model_map, 0 /*down_offset: 影子*/, 0 /*down_expert_bytes*/,
                                   in_dim, mid_dim, out_dim, selected, weights, n_total_expert, n_expert_used,
                                   clamp, x, layer, n_tok, ver);
    if (g_vqp_hdr[layer]) {                 /* 头缓存按层建过一次即可(只含偏移/维度, 不含指针) */ }
    return ok;
}
