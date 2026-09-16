/* cuda_v41_hc.inc.cu — ds4_cuda.cu 分片: DeepSeek V4.1 的 hyper-connection(mHC)一族核。
 *
 * 为什么单独一片: cuda_v41_1.inc.cu 加了合一核之后顶破了 500 行的硬规矩, 而 mHC 这一族
 * (mix → sinkhorn 拆分 → hc_pre → hc_post, 外加它自己的 rsqrt/scale_rows)本来就是一件完整的事,
 * 官方部署里它更是要合成 Mega-mHC 一个核的(报告 §3.1)。按机制命名, 不按序号。
 *
 * 口径与 cuda_v41_1 一致: f32 计算, 在官方 bf16 模块边界处显式舍 bf16; 每个核对着官方
 * model.py 的 hc_mixes / hc_split_sinkhorn / hc_pre / hc_post 那几段写。
 * ★必须在 cuda_v41_1.inc.cu 之后 include★: 用它的 v41_bf16r / v41_scratch / v41_grow / g_v41_misc。 */

/* ---- hc_mix: rsqrt(mean(flat²)+eps) 逐行, 乘在 GEMM 结果上 ---- */
__global__ static void v41_row_rsqrt_kernel(float *inv, const float *x, uint32_t dim, float eps) {
    const uint32_t r = blockIdx.x; const float *xr = x + (uint64_t)r * dim;
    float s = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    if (threadIdx.x == 0) inv[r] = rsqrtf(sh[0] / (float)dim + eps);
}
__global__ static void v41_scale_rows_kernel(float *y, const float *inv, uint32_t cols, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= inv[i / cols];
}
/* ★Mega-mHC 第一刀(decode.md D2 #1/#11, 2026-09-16): mix 的三发合一★
 *
 * 原来一次 hc_mix 发三个核: ①row_rsqrt 算每个 token 的 1/rms ②f32 GEMV 出 24 列 ③scale_rows 乘回去。
 * ①的 grid 就是 n_tok —— **解码时只有 1 个 block**, 48 个 SM 用 1 个, 读 82 KB 要 13.4 µs(合 6 GB/s),
 * 全是启动与单块延迟; 每半层一次、40 层 = 80 发/步 ≈ 1.1 ms 纯白付。③更是为做一次乘法发一个核。
 * 合成一发: GEMV 本来就有 24 个 block, 每个 block 的 256 个线程合起来正好把整条 hc 读一遍 ——
 * 让每个 block 自己再算一遍 ①(82 KB 第二遍落在 L2 里), ③并进出口那一次乘法。
 *
 * ★这两段循环不许"优化"★: ①的累加序(线程 i 走 i, i+256, …, 再 shared 树归约)与 ②的累加序
 * (warp 按 kpart 分 128 段、float4 取数、shfl 归约后按 kpart 升序相加)都是从
 * v41_row_rsqrt_kernel / v41_f32_gemv_kernel **原样抄过来的**。换一种写法就是换一个和,
 * 最后一位一变, 过一遍 sinkhorn 就放大成不同的混合系数, 整个输出就变了。
 * 出错会怎样: 不报错, 温 0 输出与合并前逐字节对不上 —— 门就是拿这个抓它。 */
__global__ static void v41_hc_mix_fused_kernel(float *mix, const float *w, const float *hc,
                                               uint32_t dim, uint32_t out_dim, float eps, uint32_t ksplit) {
    const uint32_t t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const float *x = hc + (uint64_t)t * dim;
    __shared__ float sh[256];
    /* ① 逐字照抄 v41_row_rsqrt_kernel */
    {
        float s = 0.f;
        for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += x[i] * x[i];
        sh[threadIdx.x] = s; __syncthreads();
        for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    }
    const float inv = rsqrtf(sh[0] / (float)dim + eps);
    __syncthreads();                       /* sh 下面要当归约槽复用, 等所有线程读完 sh[0] */
    /* ② 逐字照抄 v41_f32_gemv_kernel 的 ksplit 分支(这里恒是 rows_per_block=1 ⇒ r = blockIdx.x) */
    const uint32_t kpart = warp % ksplit, r = blockIdx.x;
    float acc = 0.f;
    if (r < out_dim) {
        const float *wr = w + (uint64_t)r * dim;
        for (uint32_t c = kpart * 128u + lane * 4u; c < dim; c += ksplit * 128u) {
            const float4 wv = *(const float4 *)(wr + c);
            const float4 xv = *(const float4 *)(x + c);
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0 && r < out_dim) {
        float v = 0.f;
        for (uint32_t k = 0; k < ksplit; k++) v += sh[k];
        mix[(uint64_t)t * out_dim + r] = v * inv;   /* ③ 原来那一发 scale_rows 就是这一次乘法 */
    }
}

int ds4_gpu_v41_hc_mix_tensor(ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size,
                              uint64_t fn_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!mix || !hc || hc->bytes < (uint64_t)n_tok * dim * 4 || mix->bytes < (uint64_t)n_tok * mix_hc * 4) return 0;
    /* 合一核只接"原 GEMV 恰好会选 ksplit=8 / 每 block 一行"的形状 —— 也就是现役的 24 × 20480。
     * 形状一旦不同(别的 hc 宽度/预填大批), 按原来的三发路走, 免得悄悄换一个累加序。 */
    const uint64_t wbytes = (uint64_t)dim * mix_hc * 4;
    const uint32_t nseg = dim / 128u;
    if (n_tok <= V41_GEMV_MAX_TOK && (dim % 128u) == 0u && mix_hc * 8u < 32768u && nseg >= 8u &&
        fn_offset <= model_size && wbytes <= model_size - fn_offset) {
        const float *W = (const float *)cuda_model_range_ptr(model_map, fn_offset, wbytes, "v41 hc fn");
        if (W) {
            v41_hc_mix_fused_kernel<<<dim3(mix_hc, n_tok), 256, 0, g_cur_stream>>>(
                (float *)mix->ptr, W, (const float *)hc->ptr, dim, mix_hc, eps, 8u);
            return cuda_ok(cudaGetLastError(), "v41 hc mix fused");
        }
    }
    float *inv = (float *)v41_grow(&g_v41_misc, (uint64_t)n_tok * 4, "v41 hc inv");
    if (!inv) return 0;
    v41_row_rsqrt_kernel<<<n_tok, 256, 0, g_cur_stream>>>(inv, (const float *)hc->ptr, dim, eps);
    if (!cuda_ok(cudaGetLastError(), "v41 hc rsqrt")) return 0;
    if (!ds4_gpu_v41_matmul_f32_tensor(mix, model_map, model_size, fn_offset, dim, mix_hc, hc, n_tok)) return 0;
    const uint64_t n = (uint64_t)n_tok * mix_hc;
    v41_scale_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>((float *)mix->ptr, inv, mix_hc, n);
    return cuda_ok(cudaGetLastError(), "v41 hc mix scale");
}

/* ---- sinkhorn 拆分(官方 hc_split_sinkhorn_kernel 逐式), 一 block 一行, hc ≤ 8 ---- */
__global__ static void v41_hc_split_kernel(float *pre, float *post, float *comb, const float *mix, const float *scale,
                                           const float *base, uint32_t hc, uint32_t iters, float eps) {
    const uint32_t n = blockIdx.x, mix_hc = 2u * hc + hc * hc;
    const float *m = mix + (uint64_t)n * mix_hc;
    __shared__ float c[64];
    if (threadIdx.x < hc) {
        pre[n * hc + threadIdx.x] = 1.f / (1.f + expf(-(m[threadIdx.x] * scale[0] + base[threadIdx.x]))) + eps;
        post[n * hc + threadIdx.x] = 2.f / (1.f + expf(-(m[hc + threadIdx.x] * scale[1] + base[hc + threadIdx.x])));
    }
    if (threadIdx.x < hc * hc) c[threadIdx.x] = m[2u * hc + threadIdx.x] * scale[2] + base[2u * hc + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {   /* 4×4 矩阵的 sinkhorn 串行算, 与参考逐式同序 */
        for (uint32_t j = 0; j < hc; j++) {            /* softmax(-1) + eps */
            float mx = -INFINITY; for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
            float s = 0.f; for (uint32_t k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
            for (uint32_t k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
        }
        for (uint32_t k = 0; k < hc; k++) {            /* / (sum(-2) + eps) */
            float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k];
            for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps);
        }
        for (uint32_t it = 1; it < iters; it++) {
            for (uint32_t j = 0; j < hc; j++) { float s = 0.f; for (uint32_t k = 0; k < hc; k++) s += c[j * hc + k]; for (uint32_t k = 0; k < hc; k++) c[j * hc + k] /= (s + eps); }
            for (uint32_t k = 0; k < hc; k++) { float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k]; for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps); }
        }
    }
    __syncthreads();
    if (threadIdx.x < hc * hc) comb[(uint64_t)n * hc * hc + threadIdx.x] = c[threadIdx.x];
}
int ds4_gpu_v41_hc_split_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size,
                                uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t iters,
                                float eps, uint32_t n_tok) {
    if (!pre || !post || !comb || !mix || n_hc > 8u) return 0;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "v41 hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mix_hc * 4, "v41 hc base");
    if (!sc || !bs) return 0;
    v41_hc_split_kernel<<<n_tok, 64, 0, g_cur_stream>>>((float *)pre->ptr, (float *)post->ptr, (float *)comb->ptr,
                                                         (const float *)mix->ptr, sc, bs, n_hc, iters, eps);
    return cuda_ok(cudaGetLastError(), "v41 hc split");
}

/* hc_pre: out[n][d] = Σ_c pre[n][c]·hc[n][c][d] → bf16 */
__global__ static void v41_hc_pre_kernel(float *out, const float *hc, const float *pre, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float s = 0.f;
    for (uint32_t c = 0; c < n_hc; c++) s += pre[n * n_hc + c] * hc[((uint64_t)n * n_hc + c) * n_embd + d];
    out[(uint64_t)n * n_embd + d] = v41_bf16r(s);
}
int ds4_gpu_v41_hc_pre_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre,
                              uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out || !hc || !pre) return 0;
    v41_hc_pre_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)hc->ptr, (const float *)pre->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 hc pre");
}

/* ★2026-09-15 single.md S3: hc_split + hc_pre + rms_norm 合成一发★
 * 病(逐核时间线): 解码一步 1751 发核里有约 900 发是"读几十 KB、花 2~35 µs"的小核, 合计 10 ms ——
 * 时间几乎全在启动与排空, 不在算。这三件是其中最肥的一段: 每半层 split 35 µs + pre 1.2 + norm 8 ≈ 44 µs,
 * 80 个半层就是 3.5 ms/步, 而它们读的字节加起来还不到 0.2 GB(值 0.8 ms)。
 * 合法性: 三件是严格串行的同一条链(split 出 pre → pre 加权 hc 得 x → x 归一化得 xn), 中间量除了
 * pre/post/comb 与 x 还要给别人用(hc_post 与反修钩子), 所以照旧写回 global, 只是不再各起一发。
 * ★数值逐位同★: sinkhorn 原样(thread 0 串行同序)、hc_pre 的 c 循环同序、rms 的树形归约与
 * v41_rms_norm_kernel 同一形状(256 线程 + sh[256]) —— 每个中间量在哪一步舍 bf16 一个没动。 */
/* ★pre_in 与 pre 是两回事★: 核里 split 产的 pre 是**本层的**(给下半层用), 而 hc_pre 要用的是
 * **上一层传下来的** pre(官方 Block.forward 的 pre_mix 参数)。合核时最容易顺手用成同一个 ——
 * 不报错, 只是每层的残差混合系数整体错位一层, 输出还像模像样。 */
__global__ static void v41_hc_fused_kernel(float *pre, float *post, float *comb, float *x, float *xn,
                                           const float *mix, const float *hc, const float *pre_in,
                                           const float *scale, const float *base,
                                           const float *nw, uint32_t n_embd, uint32_t n_hc, uint32_t iters,
                                           float hc_eps, float norm_eps) {
    const uint32_t n = blockIdx.x, mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *m = mix + (uint64_t)n * mix_hc;
    __shared__ float c[64], shp[8], sh[256];
    if (threadIdx.x < n_hc) {
        pre[n * n_hc + threadIdx.x] = 1.f / (1.f + expf(-(m[threadIdx.x] * scale[0] + base[threadIdx.x]))) + hc_eps;
        shp[threadIdx.x] = pre_in[n * n_hc + threadIdx.x];   /* hc_pre 用上一层的 */
        post[n * n_hc + threadIdx.x] = 2.f / (1.f + expf(-(m[n_hc + threadIdx.x] * scale[1] + base[n_hc + threadIdx.x])));
    }
    if (threadIdx.x < n_hc * n_hc) c[threadIdx.x] = m[2u * n_hc + threadIdx.x] * scale[2] + base[2u * n_hc + threadIdx.x];
    __syncthreads();
    /* ★sinkhorn 并行化(single.md S3)★: 原来 20 轮全在 thread 0 上串行, 一发 35 µs × 80 半层 = 2.8 ms/步,
     * 而它算的只是一个 4×4 矩阵。改成一个线程管一个元素(hc² ≤ 64 个线程), 行和/列和各自在**线程内**
     * 按 j/k 升序串行累加 —— ★这样加法次序与串行版逐元素相同, 数值逐位同★(换成 shfl 树形归约就不是了)。 */
    const uint32_t tj = threadIdx.x / n_hc, tk = threadIdx.x % n_hc;
    const bool act = threadIdx.x < n_hc * n_hc;
    /* ★__syncthreads 必须在分支外★(2026-09-15 实撞): 写成 `if (act) {… __syncthreads(); } else __syncthreads();`
     * 看着两边都到了 barrier, 实际是**两个不同的 barrier 点** —— 核直接挂死, GPU 不返回, 进程要 kill -9。
     * 正确形状: 算 → 全体 sync → 写 → 全体 sync。 */
    float nv = 0.f;
    if (act) {   /* ① 行 softmax + eps */
        float mx = -INFINITY;
        for (uint32_t k = 0; k < n_hc; k++) mx = fmaxf(mx, c[tj * n_hc + k]);
        float sum = 0.f;
        for (uint32_t k = 0; k < n_hc; k++) sum += expf(c[tj * n_hc + k] - mx);
        nv = expf(c[tj * n_hc + tk] - mx) / sum + hc_eps;
    }
    __syncthreads();
    if (act) c[tj * n_hc + tk] = nv;
    __syncthreads();
    if (act) {   /* ② 列归一化 */
        float cs = 0.f;
        for (uint32_t j = 0; j < n_hc; j++) cs += c[j * n_hc + tk];
        nv = c[tj * n_hc + tk] / (cs + hc_eps);
    }
    __syncthreads();
    if (act) c[tj * n_hc + tk] = nv;
    __syncthreads();
    for (uint32_t it = 1; it < iters; it++) {
        if (act) {   /* 行归一化 */
            float rs = 0.f;
            for (uint32_t k = 0; k < n_hc; k++) rs += c[tj * n_hc + k];
            nv = c[tj * n_hc + tk] / (rs + hc_eps);
        }
        __syncthreads();
        if (act) c[tj * n_hc + tk] = nv;
        __syncthreads();
        if (act) {   /* 列归一化 */
            float cs = 0.f;
            for (uint32_t j = 0; j < n_hc; j++) cs += c[j * n_hc + tk];
            nv = c[tj * n_hc + tk] / (cs + hc_eps);
        }
        __syncthreads();
        if (act) c[tj * n_hc + tk] = nv;
        __syncthreads();
    }
    if (act) comb[(uint64_t)n * n_hc * n_hc + threadIdx.x] = c[threadIdx.x];
    __syncthreads();
    /* hc_pre → x, 顺手攒 x² 给 rms(与 v41_rms_norm_kernel 同一归约形状 ⇒ 逐位同) */
    float acc = 0.f;
    const float *hcn = hc + (uint64_t)n * n_hc * n_embd;
    float *xr = x + (uint64_t)n * n_embd;
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t k = 0; k < n_hc; k++) v += shp[k] * hcn[(uint64_t)k * n_embd + d];
        v = v41_bf16r(v);
        xr[d] = v;
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)n_embd + norm_eps);
    float *on = xn + (uint64_t)n * n_embd;
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) on[d] = v41_bf16r(nw[d] * (xr[d] * inv));
}
int ds4_gpu_v41_hc_fused_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                ds4_gpu_tensor *x, ds4_gpu_tensor *xn, const ds4_gpu_tensor *mix,
                                const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre_in,
                                const void *model_map, uint64_t model_size,
                                uint64_t scale_offset, uint64_t base_offset, uint64_t norm_offset,
                                uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps,
                                uint32_t n_tok) {
    if (!pre || !post || !comb || !x || !xn || !mix || !hc || !pre_in || n_hc > 8u) return 0;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "v41 hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mix_hc * 4, "v41 hc base");
    const float *nw = (const float *)cuda_model_range_ptr(model_map, norm_offset, (uint64_t)n_embd * 4, "v41 norm w");
    if (!sc || !bs || !nw) return 0;
    v41_hc_fused_kernel<<<n_tok, 256, 0, g_cur_stream>>>((float *)pre->ptr, (float *)post->ptr, (float *)comb->ptr,
        (float *)x->ptr, (float *)xn->ptr, (const float *)mix->ptr, (const float *)hc->ptr,
        (const float *)pre_in->ptr, sc, bs, nw,
        n_embd, n_hc, iters, hc_eps, norm_eps);
    return cuda_ok(cudaGetLastError(), "v41 hc fused");
}

/* hc_post(官方 hc_post 逐式): out[k][d] = post[k]·y[d] + Σ_j comb[j][k]·res[j][d] → bf16 */
__global__ static void v41_hc_post_kernel(float *out, const float *y, const float *res, const float *post, const float *comb,
                                          uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float yv = y[(uint64_t)n * n_embd + d];
    for (uint32_t k = 0; k < n_hc; k++) {
        float s = post[n * n_hc + k] * yv;
        for (uint32_t j = 0; j < n_hc; j++) s += comb[(uint64_t)n * n_hc * n_hc + j * n_hc + k] * res[((uint64_t)n * n_hc + j) * n_embd + d];
        out[((uint64_t)n * n_hc + k) * n_embd + d] = v41_bf16r(s);
    }
}
int ds4_gpu_v41_hc_post_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res,
                               const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                               uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!out_hc || !y || !res || !post || !comb || out_hc->ptr == res->ptr) return 0;   /* 不许原地: 每 k 读全部 j */
    v41_hc_post_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, g_cur_stream>>>((float *)out_hc->ptr, (const float *)y->ptr, (const float *)res->ptr,
                                                                                      (const float *)post->ptr, (const float *)comb->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 hc post");
}
