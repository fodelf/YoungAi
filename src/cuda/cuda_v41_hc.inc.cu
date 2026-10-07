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
 * ★这两段循环的**形状**不许乱动, 但线程数是可以调的(2026-09-17)★: ①是"线程 i 走 i, i+blockDim, …
 * 再 shared 树归约", ②是"warp 按 kpart 分 128 元素段、float4 取数、shfl 归约后按 kpart 升序相加" ——
 * 形状都是从 v41_row_rsqrt_kernel / v41_f32_gemv_kernel 抄的。**线程数一变和就变**(槽数/段数都变),
 * 所以 09-15 合并那一版的门是"与三发版逐字节同", 而 09-17 把线程数从 256 提到 1024 那一刀
 * 的门只能是 NLL/top-1 —— 最后一位一变, 过一遍 sinkhorn 会放大成不同的混合系数。
 * 出错会怎样: 不报错, 温 0 输出悄悄变一个 token, 只有质量尺或逐字节比抓得到。 */
/* ★2026-09-17: 一 block 从 8 个 warp 加到 32 个(mtp-2.md §12.2 之外的一刀, ncu 定的)★
 *
 * ncu 实测(解码, 一发): **issue_active 只有 1.49%**, long_scoreboard **119~131**(专家核才 7~20),
 * warps_active 16.6%。读法: 发射槽 98.5% 是空的, warp 全在等全局访存返回 —— 不是带宽不够,
 * 是**并行度不够**。grid 只有 out_dim(=24) 个 block, 撒在 48 个 SM 上每 SM 半个 block = 4 个 warp,
 * 而一次全局访存要 ~400 个周期, 4 个 warp 的在飞请求根本盖不住。
 * 这个核一步发 80 次(每半层一次), 合 2.12 ms/步, 而它读的字节(W 1.97 MB × 80)按 240 GB/s 只该 0.65 ms。
 * ⇒ 把 block 开到 1024 个线程(32 warp, 每 SM 一个 block 就有 32 个 warp 在飞), ksplit 跟着从 8 提到 32。
 * ★数值不是逐位同★: ①的 rms 树形归约从 256 槽变 1024 槽、②的段和从 8 项变 32 项, 两处加法次序都变了
 * ⇒ 门是 NLL/top-1 质量尺, 不是 cmp。(这与"照抄累加序"那条注释冲突, 所以那段话改在下面。) */
/* ★2026-09-18: 一行拆给 V41_HC_MIX_SPLIT 个 block(解码整步 graph 落地后的小核合并)★
 * 病: grid 只有 24 个 block, 48 个 SM 一半空着; 一发 15 µs 读 1.97 MB = 131 GB/s(墙的 55%), 一步 80 发 = 1.2 ms。
 * 改: 每行的 32 个 K 段(每 warp 一段)分给 SPLIT 个 block, 每个 block 只跑自己那几段的 warp(其余 warp 在 ② 空转),
 * 段和写进 part[], **最后到的那个 block** 按段号升序把 32 项加起来(threadFenceReduction 那套: 原子计数 + fence)。
 * ★逐位同★: 每个 warp 算的段一个字没变(同 kpart 同顺序), 最后那 32 项还是按 0..31 升序加; rms 前奏每个 block 各算一遍,
 * 形状与原来一样(1024 线程走 i, i+1024, … 再 1024 槽树归约), 所以 inv 逐位同。
 * 计数器由最后那个 block 归零, 首次分配时清零一次(发射端)。 */
/* ★验证批: token 循环挪进 block(2026-09-24)★
 * 病: 原来 grid = (48, n_tok), 1024 线程的 block 每 SM 只挂一个 ⇒ 验 4 行就是 4 波串行, 同一行 W 每个 token 各读一遍;
 * 直发实测一发 n=1 26 µs / n=4 32 µs, 验证批 80 发 2.57 ms 对纯解码 1.16。
 * 改: grid 只按行 × 段(48 个 block), 每个 block 把 W 段读进寄存器一次, 再按 t 升序对每个 token 做同一套 ①②③。
 * ★逐位同★: 每个 token 的 rms 归约形状、段点积次序、32 项段和升序相加都与原来一字不差, 只是换了哪个 block 在什么时候算;
 * n=1 就是原核。门 = 投机 == 纯解码逐字节(同轨)与纯解码输出 cmp。 */
#define V41_HC_MIX_SPLIT 2u
__global__ static void v41_hc_mix_fused_kernel(float *mix, float *part, uint32_t *ctr, const float *w, const float *hc,
                                               uint32_t dim, uint32_t out_dim, float eps, uint32_t ksplit, uint32_t nsplit, uint32_t n_tok) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x / nsplit, sp = blockIdx.x % nsplit, kper = ksplit / nsplit;
    __shared__ float sh[1024];
    __shared__ uint32_t s_last;
    /* ★W 先发, 再做 ①(2026-09-23)★: 原来 ① 整段(读 82 KB hc + 1024 槽树归约 + 十几次全体屏障)做完才开始读 W, 每发 16.5 µs
     * 读 1.97 MB = 119 GB/s。W 与 ① 无依赖 ⇒ 进核就把本 warp 的 W 段读进寄存器, DRAM 延迟藏在 ① 后面。
     * ★逐位同★: ② 每一项仍是同一个 float4 点积、按 c 升序进 acc; 只是 W 早读了。现役 dim 20480 / 段跨 4096 ⇒ 每 lane 恰 5 段 = PF
     * (1024 线程 ⇒ 寄存器上限 64, PF 取 8 会和 ① 的一批 10 个抢寄存器)。 */
    constexpr uint32_t PF = 5u;
    const uint32_t kpart = sp * kper + warp;
    const bool act = r < out_dim && warp < kper;
    const float *wr = w + (uint64_t)r * dim;
    const uint32_t c0 = kpart * 128u + lane * 4u, cst = ksplit * 128u;
    float4 wpre[PF];
    #pragma unroll
    for (uint32_t i = 0; i < PF; i++) {
        wpre[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (act && c0 + i * cst < dim) wpre[i] = *(const float4 *)(wr + c0 + i * cst);
    }
    v41_pdl_wait();   /* PDL: W(常量)已在飞, 这里才等上游的 hc(见 cuda_internal.cuh) */
    for (uint32_t t = 0; t < n_tok; t++) {   /* 块内按 token 升序; 下面整段与单 token 版逐字同, 只多一层缩进的循环 */
    const float *x = hc + (uint64_t)t * dim;
    /* ① 逐字照抄 v41_row_rsqrt_kernel 的累加形状; ★读法改成 10 个一批先发(2026-09-23)★: 原循环边界是运行期值, 编译器
     * 不展开, 每线程 20 次 L2 读一次一个往返串行排队(~6 µs)。现在一批 10 个一起发、再按 i 升序加 ⇒ 累加序不变, 逐位同。 */
    {
        float s = 0.f;
        for (uint32_t i0 = threadIdx.x; i0 < dim; i0 += 10u * blockDim.x) {
            float xv[10];
            #pragma unroll
            for (uint32_t j = 0; j < 10u; j++) { const uint32_t i = i0 + j * blockDim.x; xv[j] = i < dim ? x[i] : 0.f; }
            #pragma unroll
            for (uint32_t j = 0; j < 10u; j++) if (i0 + j * blockDim.x < dim) s += xv[j] * xv[j];
        }
        sh[threadIdx.x] = s; __syncthreads();
        /* ★树的尾巴进 warp 0(2026-10-07)★: 原来 10 级每级一次全体屏障(1024 线程一次 ~0.5 µs), 一发 4 个 token 串行就是 40 次屏障, 占这发 36.9 µs 的大头
         * (验证步 80 发 = 1.43 ms 独占)。k ≥ 64 的四级照旧(跨 warp 要屏障); k=32 起只剩 warp 0 的 32 个槽: 它自己从 shared 取 sh[i+32] 加到 sh[i],
         * 再用 shfl_down 做 k=16..1 —— **每一级加的仍是 sh[i] + sh[i+k] 这两个数、同一个左右次序**, 只是不再写回 shared, 所以和逐位同;
         * 上半 lane 的值没人用。屏障 11 → 6 次/token。门 = 温 0 输出 cmp(同轨 + 纯解码)。 */
        for (uint32_t k = blockDim.x / 2; k > 32u; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
        if (threadIdx.x < 32u) {
            float v = sh[threadIdx.x] + sh[threadIdx.x + 32u];              /* k=32 */
            #pragma unroll
            for (int k = 16; k > 0; k >>= 1) v += __shfl_down_sync(0xffffffffu, v, k);   /* k=16..1: lane i 得 sh[i] + sh[i+k] */
            if (threadIdx.x == 0) sh[0] = v;
        }
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)dim + eps);
    /* ② 逐字照抄 v41_f32_gemv_kernel 的 ksplit 分支(这里恒是 rows_per_block=1); 本 block 只管 kpart ∈ [sp·kper, sp·kper+kper) */
    float acc = 0.f;
    if (act) {   /* warp 内一致的分支, shfl 全 mask 安全 */
        #pragma unroll
        for (uint32_t i = 0; i < PF; i++) {   /* 前 PF 段用先发的 W, 次序与原循环相同 */
            const uint32_t c = c0 + i * cst;
            if (c < dim) {
                const float4 wv = wpre[i];
                const float4 xv = *(const float4 *)(x + c);
                acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
        for (uint32_t c = c0 + PF * cst; c < dim; c += cst) {   /* dim 更大时的剩余段(现役没有) */
            const float4 wv = *(const float4 *)(wr + c);
            const float4 xv = *(const float4 *)(x + c);
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) part[((uint64_t)t * out_dim + r) * ksplit + kpart] = acc;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(&ctr[(uint64_t)t * out_dim + r], 1u) == nsplit - 1u) ? 1u : 0u;
    __syncthreads();
    if (s_last && threadIdx.x == 0 && r < out_dim) {   /* 最后到的 block 收尾: 32 项按段号升序相加, 计数器归零 */
        __threadfence();
        const float *pr = part + ((uint64_t)t * out_dim + r) * ksplit;
        float v = 0.f;
        for (uint32_t k = 0; k < ksplit; k++) v += __ldcg(pr + k);   /* 绕过 L1 读别的 block 写的段和; ksplit 项按段号升序(确定性) */
        mix[(uint64_t)t * out_dim + r] = v * inv;   /* ③ 原来那一发 scale_rows 就是这一次乘法 */
        ctr[(uint64_t)t * out_dim + r] = 0u;
    }
    /* 下一个 t 改写 sh/s_last 之前不用再加屏障: 本轮所有线程读 sh[0](inv)与 s_last 都在上面两道 __syncthreads 之前或之间,
     * 而下一轮第一次写 sh 在它自己的 ① 里、s_last 在 fence 后那道屏障之后 —— 谁也追不上谁。 */
    }
}
static v41_scratch g_v41_hc_part, g_v41_hc_ctr;

int ds4_gpu_v41_hc_mix_tensor(ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size,
                              uint64_t fn_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    const uint32_t dim = n_embd * n_hc, mix_hc = 2u * n_hc + n_hc * n_hc;
    if (!mix || !hc || hc->bytes < (uint64_t)n_tok * dim * 4 || mix->bytes < (uint64_t)n_tok * mix_hc * 4) return 0;
    /* 合一核只接"原 GEMV 恰好会选 ksplit=8 / 每 block 一行"的形状 —— 也就是现役的 24 × 20480。
     * 形状一旦不同(别的 hc 宽度/预填大批), 按原来的三发路走, 免得悄悄换一个累加序。 */
    const uint64_t wbytes = (uint64_t)dim * mix_hc * 4;
    const uint32_t nseg = dim / 128u;
    if (n_tok <= V41_GEMV_MAX_TOK && (dim % 128u) == 0u && mix_hc * 8u < 32768u && nseg >= 32u &&
        fn_offset <= model_size && wbytes <= model_size - fn_offset) {
        const float *W = (const float *)cuda_model_range_ptr(model_map, fn_offset, wbytes, "v41 hc fn");
        if (W) {
            /* 1024 线程 = 32 warp: 见核头那段 ncu 账。★ksplit 必须等于 warp 数★(每 warp 一个 K 段),
             * 而 K 的分段单位是 128 个元素 ⇒ dim 至少要 32×128 = 4096(现役 20480 ✓)。 */
            /* 段和暂存 + 每行的到达计数器: 只在直发路首次分配(捕获态下 v41_grow 会作废捕获); 计数器分配后清零一次,
             * 之后由最后到的 block 自己归零。 */
            const uint64_t npart = (uint64_t)n_tok * mix_hc;
            const int fresh = g_v41_hc_ctr.cap < npart * 4;
            float *part = (float *)v41_grow(&g_v41_hc_part, npart * 32u * 4, "v41 hc mix part");
            uint32_t *ctr = (uint32_t *)v41_grow(&g_v41_hc_ctr, npart * 4, "v41 hc mix ctr");
            if (!part || !ctr) return 0;
            if (fresh && !cuda_ok(cudaMemsetAsync(ctr, 0, (size_t)npart * 4, g_cur_stream ? g_cur_stream : cudaStreamPerThread), "v41 hc mix ctr zero")) return 0;
            v41_pdl_register((const void *)v41_hc_mix_fused_kernel);   /* 核在碰 hc/part/ctr 之前 v41_pdl_wait */
            v41_hc_mix_fused_kernel<<<mix_hc * V41_HC_MIX_SPLIT, 1024, 0, g_cur_stream>>>(
                (float *)mix->ptr, part, ctr, W, (const float *)hc->ptr, dim, mix_hc, eps, 32u, V41_HC_MIX_SPLIT, n_tok);
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
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
    const uint32_t n = blockIdx.x, mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *m = mix + (uint64_t)n * mix_hc;
    __shared__ float c[64], sh[1024];   /* sh 的槽数 = blockDim(见发射端那段 ncu 账) */
    /* ★sinkhorn 整段只在 warp 0 里做(2026-09-18, 小核合并)★
     * 病: 它算的是一个 4×4 矩阵(16 个线程), 却让 1024 个线程陪着过 20 轮 × 4 次 __syncthreads = 80 多个全 block 屏障,
     * 一发 14.5 µs 里大半是屏障; 一步 80 发 = 1.16 ms。
     * 改: n_hc² ≤ 32 个活跃线程全在 warp 0 里, 屏障换成 __syncwarp(只同步一个 warp, 几个周期), 其余 31 个 warp 直接到
     * 下面 hc_pre 前的那一次 __syncthreads 等着。★逐位同★: 每个元素的算式、行和/列和的累加序一个字没动, 只换了同步。
     * 前提: n_hc ≤ 5(n_hc² ≤ 32); 发射端把 n_hc > 5 挡掉。 */
    const uint32_t warp = threadIdx.x >> 5;
    if (warp == 0) {
        if (threadIdx.x < n_hc) {
            pre[n * n_hc + threadIdx.x] = 1.f / (1.f + expf(-(m[threadIdx.x] * scale[0] + base[threadIdx.x]))) + hc_eps;
            post[n * n_hc + threadIdx.x] = 2.f / (1.f + expf(-(m[n_hc + threadIdx.x] * scale[1] + base[n_hc + threadIdx.x])));
        }
        if (threadIdx.x < n_hc * n_hc) c[threadIdx.x] = m[2u * n_hc + threadIdx.x] * scale[2] + base[2u * n_hc + threadIdx.x];
        __syncwarp();
        /* ★sinkhorn 并行化(single.md S3)★: 原来 20 轮全在 thread 0 上串行, 一发 35 µs × 80 半层 = 2.8 ms/步,
         * 而它算的只是一个 4×4 矩阵。改成一个线程管一个元素(hc² 个线程), 行和/列和各自在**线程内**
         * 按 j/k 升序串行累加 —— ★这样加法次序与串行版逐元素相同, 数值逐位同★(换成 shfl 树形归约就不是了)。
         * 同步点必须在分支外(2026-09-15 实撞: 分支内的屏障 = 两个不同的屏障点, 核挂死): 算 → 全体 sync → 写 → 全体 sync。 */
        const uint32_t tj = threadIdx.x / n_hc, tk = threadIdx.x % n_hc;
        const bool act = threadIdx.x < n_hc * n_hc;
        float nv = 0.f;
        if (act) {   /* ① 行 softmax + eps */
            float mx = -INFINITY;
            for (uint32_t k = 0; k < n_hc; k++) mx = fmaxf(mx, c[tj * n_hc + k]);
            float sum = 0.f;
            for (uint32_t k = 0; k < n_hc; k++) sum += expf(c[tj * n_hc + k] - mx);
            nv = expf(c[tj * n_hc + tk] - mx) / sum + hc_eps;
        }
        __syncwarp();
        if (act) c[tj * n_hc + tk] = nv;
        __syncwarp();
        if (act) {   /* ② 列归一化 */
            float cs = 0.f;
            for (uint32_t j = 0; j < n_hc; j++) cs += c[j * n_hc + tk];
            nv = c[tj * n_hc + tk] / (cs + hc_eps);
        }
        __syncwarp();
        if (act) c[tj * n_hc + tk] = nv;
        __syncwarp();
        for (uint32_t it = 1; it < iters; it++) {
            if (act) {   /* 行归一化 */
                float rs = 0.f;
                for (uint32_t k = 0; k < n_hc; k++) rs += c[tj * n_hc + k];
                nv = c[tj * n_hc + tk] / (rs + hc_eps);
            }
            __syncwarp();
            if (act) c[tj * n_hc + tk] = nv;
            __syncwarp();
            if (act) {   /* 列归一化 */
                float cs = 0.f;
                for (uint32_t j = 0; j < n_hc; j++) cs += c[j * n_hc + tk];
                nv = c[tj * n_hc + tk] / (cs + hc_eps);
            }
            __syncwarp();
            if (act) c[tj * n_hc + tk] = nv;
            __syncwarp();
        }
        if (act) comb[(uint64_t)n * n_hc * n_hc + threadIdx.x] = c[threadIdx.x];
    }
    /* ★hc_pre 不等 sinkhorn(2026-09-23)★: 它用的是**上一层**传下来的 pre_in, 与本层 warp 0 的 sinkhorn 无依赖 ——
     * 原来 warp 0 把 pre_in 抄进 shared 再全体屏障, 其余 31 个 warp 白等 20 轮 sinkhorn。现在各线程直接读 pre_in(同一个值,
     * 广播读), 屏障去掉; warp 0 算完 sinkhorn 再做自己那份 hc_pre, 下面 rms 归约前的屏障照样把大家对齐。★逐位同★: 值与次序都没变。 */
    /* hc_pre → x, 顺手攒 x² 给 rms(与 v41_rms_norm_kernel 同一归约形状 ⇒ 逐位同) */
    float acc = 0.f;
    const float *hcn = hc + (uint64_t)n * n_hc * n_embd;
    const float *pin = pre_in + (uint64_t)n * n_hc;
    float *xr = x + (uint64_t)n * n_embd;
    /* ★读法: 本线程的 5 个 d(n_embd 5120 / 1024)× n_hc 路一次全发, 再按 d 升序算(2026-09-23)★: 原循环运行期边界不展开,
     * 5 轮各等一次 L2 往返。算式与次序不变(v 按 k 升序、acc 按 d 升序), x 值留在寄存器给出口那段复用(与读回 xr[d] 同值)。
     * 只接 n_hc == 4 且 n_embd ≤ 5 × blockDim 的现役形状; 其余形状走原循环(同一算式)。 */
    constexpr uint32_t DP = 5u;
    float *on = xn + (uint64_t)n * n_embd;
    if (n_hc == 4u && n_embd <= DP * blockDim.x) {
        float hv[DP][4], vv[DP], wv[DP];
        #pragma unroll
        for (uint32_t j = 0; j < DP; j++) {
            const uint32_t d = threadIdx.x + j * blockDim.x;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; k++) hv[j][k] = d < n_embd ? hcn[(uint64_t)k * n_embd + d] : 0.f;
            wv[j] = d < n_embd ? nw[d] : 0.f;
        }
        const float p0 = pin[0], p1 = pin[1], p2 = pin[2], p3 = pin[3];
        #pragma unroll
        for (uint32_t j = 0; j < DP; j++) {
            const uint32_t d = threadIdx.x + j * blockDim.x;
            if (d < n_embd) {
                float v = 0.f;
                v += p0 * hv[j][0]; v += p1 * hv[j][1]; v += p2 * hv[j][2]; v += p3 * hv[j][3];
                v = v41_bf16r(v);
                xr[d] = v; vv[j] = v;
                acc += v * v;
            }
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
        const float inv = rsqrtf(sh[0] / (float)n_embd + norm_eps);
        #pragma unroll
        for (uint32_t j = 0; j < DP; j++) { const uint32_t d = threadIdx.x + j * blockDim.x; if (d < n_embd) on[d] = v41_bf16r(wv[j] * (vv[j] * inv)); }
        return;
    }
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float v = 0.f;
        for (uint32_t k = 0; k < n_hc; k++) v += pin[k] * hcn[(uint64_t)k * n_embd + d];
        v = v41_bf16r(v);
        xr[d] = v;
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)n_embd + norm_eps);
    for (uint32_t d = threadIdx.x; d < n_embd; d += blockDim.x) on[d] = v41_bf16r(nw[d] * (xr[d] * inv));
}
int ds4_gpu_v41_hc_fused_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                ds4_gpu_tensor *x, ds4_gpu_tensor *xn, const ds4_gpu_tensor *mix,
                                const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre_in,
                                const void *model_map, uint64_t model_size,
                                uint64_t scale_offset, uint64_t base_offset, uint64_t norm_offset,
                                uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps,
                                uint32_t n_tok) {
    if (!pre || !post || !comb || !x || !xn || !mix || !hc || !pre_in || n_hc > 5u) return 0;   /* n_hc² 要塞进一个 warp(见核里 sinkhorn 那段) */
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *sc = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12, "v41 hc scale");
    const float *bs = (const float *)cuda_model_range_ptr(model_map, base_offset, (uint64_t)mix_hc * 4, "v41 hc base");
    const float *nw = (const float *)cuda_model_range_ptr(model_map, norm_offset, (uint64_t)n_embd * 4, "v41 norm w");
    if (!sc || !bs || !nw) return 0;
    /* ★1024 线程(2026-09-17, ncu 定的)★: 这个核的 grid 就是 n_tok —— 解码时**只有一个 block**,
     * 48 个 SM 里 47 个空转。ncu: issue_active 7.8%, warps_active 16.6%, long_scoreboard 15.9 ——
     * 不是带宽也不是算力, 是"8 个 warp 的指令依赖链填不满一个 SM"。一步发 80 次 = 1.75 ms/步,
     * 而它读的字节(hc 80 KB × 80)按 240 GB/s 只该 0.03 ms。
     * 线程开到 1024(32 warp)后 hc_pre 那 5120 个输出维度每轮覆盖 1024 个(原来 256), 访存并行度 4 倍。
     * ★数值不是逐位同★: rms 的树形归约从 256 槽变 1024 槽 ⇒ 加法次序变 ⇒ 门是 NLL/top-1。
     * (sinkhorn 那段只用 n_hc² ≤ 64 个线程, 其余空转但要一起过 barrier —— 它本来就是这个形状。) */
    v41_hc_fused_kernel<<<n_tok, 1024, 0, g_cur_stream>>>((float *)pre->ptr, (float *)post->ptr, (float *)comb->ptr,
        (float *)x->ptr, (float *)xn->ptr, (const float *)mix->ptr, (const float *)hc->ptr,
        (const float *)pre_in->ptr, sc, bs, nw,
        n_embd, n_hc, iters, hc_eps, norm_eps);
    return cuda_ok(cudaGetLastError(), "v41 hc fused");
}

/* hc_post(官方 hc_post 逐式): out[k][d] = post[k]·y[d] + Σ_j comb[j][k]·res[j][d] → bf16 */
__global__ static void v41_hc_post_kernel(float *out, const float *y, const float *res, const float *post, const float *comb,
                                          uint32_t n_embd, uint32_t n_hc) {
    v41_pdl_wait();   /* PDL: 第一句就等上游(见 cuda_internal.cuh); 不经 PDL 发射时立即返回 */
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
