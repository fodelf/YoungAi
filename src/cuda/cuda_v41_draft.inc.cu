/* cuda_v41_draft.inc.cu — ds4_cuda.cu 分片: DSpark 草稿塔(speed.md 段 6 D1)专用核。
 *
 * 草稿塔与普通层的差别只有两处, 其余(hc/norm/注意力/shared 专家/出口)全复用主路的核:
 *   ① 专家不是 VQ blob 而是**逐专家一个 fp4x32 张量**(speed.md §2: "草稿质量 = 接受率 = 速度, 不省这里"),
 *      所以要一条按 sel 间接寻址的 dense MoE —— 权重基址由路由选出来的专家号决定, 不能像骨架那样编译期定。
 *   ② 主模型的注意力输入(hc 四路均值)要在主前向里顺手取走, 拼成 main_hidden ⇒ 一个均值搬运核。
 * 另外 markov 头要按"设备上刚采出来的 token id"去查 bf16 表的一行(主机不参与, 不然每位一次往返)。
 *
 * 出错会怎样: main_hidden 取错位置(取成层输出而不是层的注意力输入)不报错, 只是接受率掉到 1 附近
 * (speed.md §7 点名的坑); dense MoE 的 sel 越界会读到别的专家的字节, 也不报错, 所以核里对 e 做了范围判。 */

/* 一 warp 算 fp4x32 权重的一行与激活 x 的点积(f32 累加, 激活按 bf16 格点读)。
 * 排布与 cuda_v41_4 的骨架 GEMV 同: 4 个 lane 分一个 32 元素块, 每 lane 拿 4 个连续字节 = 8 个元素
 * (scale 一块共用一次解码, 摊到 8 个元素上)。返回值只有 lane 0..31 规约后的全和。 */
__device__ __forceinline__ static float v41_fp4_row_dot(const uint8_t *wr, const float *x, uint32_t nblk) {
    const uint32_t lane = threadIdx.x & 31u, q = lane & 3u, sub = lane >> 2;
    float acc = 0.f;
    const uint32_t ngrp = (nblk + 7u) / 8u;
    for (uint32_t gi = 0; gi < ngrp; gi++) {
        const uint32_t b = gi * 8u + sub;
        if (b >= nblk) continue;   /* 尾组空位: 别的 lane 还有活, 不能 break */
        const uint8_t *p = wr + (uint64_t)b * 17u;
        const float sc = ds4_e8m0_to_f32(p[16]);
        const float *xb = x + b * 32u + q * 8u;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint8_t by = p[q * 4u + j];
            acc += ds4_fp4_nibble_to_f32(by & 0x0Fu) * sc * v41_bf16r(xb[2u * j]);
            acc += ds4_fp4_nibble_to_f32(by >> 4)    * sc * v41_bf16r(xb[2u * j + 1u]);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

/* gate/up 同核(官方 Expert: w1/w3 出 bf16 → swiglu → bf16)。一 warp 一行, 一 block 8 行。
 * grid (ceil(MID/8), n_tok·K): blockIdx.y 是 (token, 第 k 个选中专家) 这一对。 */
__global__ static void v41_mtp_gateup_kernel(float *h, const uint8_t *const *wg, const uint8_t *const *wu,
                                             const int32_t *sel, const float *x, uint32_t IN, uint32_t MID,
                                             uint32_t K, uint32_t n_expert, float clamp) {
    const uint32_t pair = blockIdx.y, t = pair / K, warp = threadIdx.x >> 5;
    const int32_t e = sel[pair];
    if (e < 0 || (uint32_t)e >= n_expert) return;
    const uint32_t r = blockIdx.x * 8u + warp;
    if (r >= MID) return;
    const uint32_t nblk = IN / 32u;
    const float *xs = x + (uint64_t)t * IN;
    float gv = v41_bf16r(v41_fp4_row_dot(wg[e] + (uint64_t)r * nblk * 17u, xs, nblk));
    float uv = v41_bf16r(v41_fp4_row_dot(wu[e] + (uint64_t)r * nblk * 17u, xs, nblk));
    if (clamp > 0.f) { if (gv > clamp) gv = clamp; if (uv > clamp) uv = clamp; if (uv < -clamp) uv = -clamp; }
    if ((threadIdx.x & 31u) == 0) h[(uint64_t)pair * MID + r] = v41_bf16r((gv / (1.0f + expf(-gv))) * uv);
}

/* down: partial[pair][OUT] = bf16(W2·h) */
__global__ static void v41_mtp_down_kernel(float *partial, const uint8_t *const *wd, const int32_t *sel,
                                           const float *h, uint32_t MID, uint32_t OUT, uint32_t K, uint32_t n_expert) {
    const uint32_t pair = blockIdx.y, warp = threadIdx.x >> 5;
    const int32_t e = sel[pair];
    const uint32_t r = blockIdx.x * 8u + warp;
    if (r >= OUT) return;
    if (e < 0 || (uint32_t)e >= n_expert) {   /* 没选中的槽要写 0: 下游 reduce 会把整行读进去 */
        if ((threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = 0.f;
        return;
    }
    const uint32_t nblk = MID / 32u;
    const float y = v41_fp4_row_dot(wd[e] + (uint64_t)r * nblk * 17u, h + (uint64_t)pair * MID, nblk);
    if ((threadIdx.x & 31u) == 0) partial[(uint64_t)pair * OUT + r] = v41_bf16r(y);
}

/* 每塔的专家权重设备指针表: 128 个专家 × 3 个矩阵, 解析一次存着。
 * 为什么要这张表: 核里要按 sel 选基址, 而 GGUF 里每个专家是一个独立张量(偏移不连续),
 * 每步现解 384 次 range 查找是白花的主机时间。 */
typedef struct { const uint8_t **dev_g, **dev_u, **dev_d; uint32_t n; } v41_mtp_ptrs;
static v41_mtp_ptrs g_mtp_ptrs[8];   /* 塔数上限(官方 3); 只是这张表的槽位, 真值由调用方给 */
static v41_scratch g_mtp_h, g_mtp_part;

static int v41_mtp_bind(uint32_t tower, const void *model_map, const uint64_t *off, uint32_t n_expert) {
    if (tower >= 8u) return 0;
    v41_mtp_ptrs *P = &g_mtp_ptrs[tower];
    if (P->n == n_expert && P->dev_g) return 1;
    const uint8_t **host = (const uint8_t **)malloc(sizeof(void *) * 3u * n_expert);
    if (!host) return 0;
    for (uint32_t i = 0; i < 3u * n_expert; i++) {
        /* bytes 给 1: 这里只要基址, 越界与否由核里的行号算法与 GGUF 登记保证(行数 × 块数 × 17 B) */
        const char *p = cuda_model_range_ptr(model_map, off[i], 1, "mtp expert");
        if (!p) { free((void *)host); return 0; }
        host[i] = (const uint8_t *)p;
    }
    void *dev = NULL;
    if (cudaMalloc(&dev, sizeof(void *) * 3u * n_expert) != cudaSuccess) { (void)cudaGetLastError(); free((void *)host); return 0; }
    const cudaError_t st = cudaMemcpy(dev, host, sizeof(void *) * 3u * n_expert, cudaMemcpyHostToDevice);
    free((void *)host);
    if (st != cudaSuccess) { (void)cudaGetLastError(); (void)cudaFree(dev); return 0; }
    P->dev_g = (const uint8_t **)dev;
    P->dev_u = P->dev_g + n_expert;
    P->dev_d = P->dev_g + 2u * n_expert;
    P->n = n_expert;
    return 1;
}

int ds4_gpu_v41_mtp_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint32_t tower, const uint64_t *exp_off,
                               uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                               const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                               uint32_t n_expert, uint32_t topk, float clamp,
                               const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !selected || !weights || !x || !exp_off || !n_tok || !topk) return 0;
    if ((in_dim % 32u) || (mid_dim % 32u)) return 0;
    if (!v41_mtp_bind(tower, model_map, exp_off, n_expert)) return 0;
    const v41_mtp_ptrs *P = &g_mtp_ptrs[tower];
    const uint64_t np = (uint64_t)n_tok * topk;
    float *h = (float *)v41_grow(&g_mtp_h, np * mid_dim * 4, "mtp h");
    float *part = (float *)v41_grow(&g_mtp_part, np * out_dim * 4, "mtp partial");
    if (!h || !part) return 0;
    /* ★2026-09-17 判负存档: 草稿塔按专家并集(mtp-2.md §6 刀 1)★
     * 依据看着很硬: `[mtp-uniq]` 实测一塔 15 个 (位,专家) 对里只有 **6.67 个唯一专家(44%)** —— 块里第 1..4 位
     * 吃同一个 noise 嵌入, 路由几乎一样; 而这个核 0.79 GB / 4.43 ms = 178 GB/s 看着像"贴着带宽墙",
     * 贴墙就该"省 56% 字节 = 省 56% 时间"。写了一版一 block 管一个专家、组里成员数 CNT 做模板参数
     * (运行期下标会让 acc 掉进 local memory, 09-15 撞过), 权重读一遍、对组里每条激活各乘一遍; 逐位同
     * (接受率直方图一字不差)。**实测草稿 13.4 → 13.1 ms(2K) / 13.6 → 13.5(12k), 0.3%, 噪声内。**
     * ★真因: 178 GB/s 的"贴墙"是巧合, 这个核其实是波前受限★ —— 一轮 32 个 lane 取 4 B 权重字节 = 1 个波前,
     * 取 16 B 激活 = 4 个波前, **激活占 4/5**。并集省掉的是权重那 1 个, 激活那 4 个还得按成员数乘,
     * 于是每对从 5 个波前降到 4.33(省 13%), 再被 CNT 倍的 FFMA 与地址算术吃掉大半。
     * ⇒ 与主干 VQ 那两次去重判负(09-16/09-17)**最终同源**: 这一族核的瓶颈都是"每个 token 各自那份激活读",
     * 不是权重字节。别再从"省权重字节"这个方向来了(这是第 8 次)。 */
    v41_mtp_gateup_kernel<<<dim3((mid_dim + 7u) / 8u, (unsigned)np), 256, 0, g_cur_stream>>>(
        h, P->dev_g, P->dev_u, (const int32_t *)selected->ptr, (const float *)x->ptr, in_dim, mid_dim, topk, n_expert, clamp);
    if (!cuda_ok(cudaGetLastError(), "v41 mtp gateup")) return 0;
    v41_mtp_down_kernel<<<dim3((out_dim + 7u) / 8u, (unsigned)np), 256, 0, g_cur_stream>>>(
        part, P->dev_d, (const int32_t *)selected->ptr, h, mid_dim, out_dim, topk, n_expert);
    if (!cuda_ok(cudaGetLastError(), "v41 mtp down")) return 0;
    v41_vq_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tok), 256, 0, g_cur_stream>>>(
        (float *)out->ptr, part, (const float *)weights->ptr, topk, out_dim);
    return cuda_ok(cudaGetLastError(), "v41 mtp reduce");
}

/* hc 四路均值 → main_hidden 的第 slot 段(官方 Transformer.forward: main_hiddens.append(h.mean(dim=2)))。
 * ★取的是 engram 之后、层之前的 h★ —— 调用点在 core_v41_forward.c 的层循环里, 不是层输出。 */
/* ★只搬最后 n_rows 行★: 预填一块有几千个 token, 而草稿器只吃"最后那几位"的 main_hidden
 * (下一轮草稿从最后一个已定 token 出发) —— 整块都搬的话 4096 × 3 × 5120 × 4 B = 251 MB 白写。 */
/* ★落点是按绝对位置定格的环★(2026-09-18, 见 ds4_gpu_v41.h 的声明注释): 第 t 行 → ((pos + t) % cap) 格。
 * graph 路 n=1、位置只在设备槽: pos = posd[0] + src_row0(src_row0 恒 0)。 */
__global__ static void v41_hc_mean_kernel(float *out, const float *hc, uint32_t E, uint32_t n_hc, uint32_t n_rows,
                                          uint32_t src_row0, uint32_t slot, uint32_t n_slot,
                                          uint32_t dst_pos0, uint32_t cap, const int32_t *posd) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_rows * E) return;
    const uint32_t t = (uint32_t)(i / E), d = (uint32_t)(i % E);
    const float *h = hc + (uint64_t)(src_row0 + t) * n_hc * E + d;
    float s = 0.f;
    for (uint32_t c = 0; c < n_hc; c++) s += h[(uint64_t)c * E];
    const uint32_t pos = posd ? (uint32_t)posd[0] + src_row0 : dst_pos0;
    out[(uint64_t)((pos + t) % cap) * n_slot * E + (uint64_t)slot * E + d] = s / (float)n_hc;
}
int ds4_gpu_v41_hc_mean_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, uint32_t n_embd, uint32_t n_hc,
                               uint32_t n_rows, uint32_t src_row0, uint32_t slot, uint32_t n_slot,
                               uint32_t dst_pos0, uint32_t cap, const ds4_gpu_tensor *posd) {
    if (!out || !hc || !n_rows || !cap || n_rows > cap) return 0;
    const uint64_t n = (uint64_t)n_rows * n_embd;
    if (out->bytes < (uint64_t)cap * n_slot * n_embd * 4) return 0;
    v41_hc_mean_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>(
        (float *)out->ptr, (const float *)hc->ptr, n_embd, n_hc, n_rows, src_row0, slot, n_slot,
        dst_pos0, cap, posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 hc mean");
}

/* 环 → 连续: dst[t] = ring[(first_row + t) % cap]。一 block 一行, 行内按 float 步进(row_floats = 目标层数 × E = 15360) */
/* firstd(2026-09-22, 草稿一轮进图): 非 NULL 时起始行从设备 int 读(图开头由零拷贝小核灌), 主机的 first_row 不用 —— 每轮补窗口的起点
 * 随上一轮接受数变, 烤进图就错(不报错, 只是窗口里的行错位, 接受率掉)。 */
__global__ static void v41_ring_rows_kernel(float *dst, const float *ring, uint32_t row_floats, uint32_t cap,
                                            uint32_t first_row, uint32_t count, const int32_t *firstd) {
    const uint32_t t = blockIdx.x;
    if (t >= count) return;
    if (firstd) first_row = (uint32_t)firstd[0];
    const float *src = ring + (uint64_t)((first_row + t) % cap) * row_floats;
    float *d = dst + (uint64_t)t * row_floats;
    for (uint32_t j = threadIdx.x; j < row_floats; j += blockDim.x) d[j] = src[j];
}
int ds4_gpu_v41_ring_rows_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *ring, uint32_t row_floats, uint32_t cap,
                                 uint32_t first_row, uint32_t count, const ds4_gpu_tensor *firstd) {
    if (!dst || !ring || !count || !cap || count > cap) return 0;
    if (dst->bytes < (uint64_t)count * row_floats * 4 || ring->bytes < (uint64_t)cap * row_floats * 4) return 0;
    v41_ring_rows_kernel<<<count, 256, 0, g_cur_stream>>>((float *)dst->ptr, (const float *)ring->ptr, row_floats, cap, first_row, count,
                                                          firstd ? (const int32_t *)firstd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 ring rows");
}

/* markov 头的 embed: 按**设备上**的 token id 取 bf16 表的一行 → f32(bf16 格点)。
 * 为什么不读回主机: 草稿块 5 位是逐位依赖的(第 i 位采出来才能查第 i 位的偏置), 每位一次 D2H
 * 就是每轮 5 次停等 —— 整条投机路的收益还不够付这个。 */
__global__ static void v41_row_gather_kernel(float *out, const uint8_t *tab, const int32_t *ids,
                                             uint32_t which, uint32_t dim, uint32_t out_row, uint32_t is_f32) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const int32_t id = ids[which];
    float v = 0.f;
    if (id >= 0) {
        if (is_f32) memcpy(&v, tab + ((uint64_t)id * dim + d) * 4u, 4);
        else { __nv_bfloat16 h; memcpy(&h, tab + ((uint64_t)id * dim + d) * 2u, 2); v = __bfloat162float(h); }
    }
    out[(uint64_t)out_row * dim + d] = v;
}
/* elem_bytes 由调用方按 GGUF 登记的张量类型给(4 = f32, 2 = bf16) —— 引擎不假设盘上是哪一种,
 * 猜错不报错, 只会读出一片垃圾(实撞过: markov 头按 bf16 读 f32 的表, 接受率 0.14/5)。 */
int ds4_gpu_v41_row_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t tab_offset, uint64_t n_rows, uint32_t dim, uint32_t elem_bytes,
                                  const ds4_gpu_tensor *ids, uint32_t which, uint32_t out_row) {
    if (!out || !ids || (elem_bytes != 2u && elem_bytes != 4u)) return 0;
    const uint64_t bytes = n_rows * dim * elem_bytes;
    if (tab_offset > model_size || bytes > model_size - tab_offset) return 0;
    const uint8_t *tab = (const uint8_t *)cuda_model_range_ptr(model_map, tab_offset, bytes, "markov embed");
    if (!tab) return 0;
    v41_row_gather_kernel<<<(dim + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (float *)out->ptr, tab, (const int32_t *)ids->ptr, which, dim, out_row, elem_bytes == 4u);
    return cuda_ok(cudaGetLastError(), "v41 row gather");
}

/* logits 的第 row 行 += bias 行(markov 偏置, 官方 logits[:, i].add_(logits_bias)) */
__global__ static void v41_row_add_kernel(float *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
int ds4_gpu_v41_row_add_tensor(ds4_gpu_tensor *dst, uint64_t dst_row, const ds4_gpu_tensor *src, uint64_t n) {
    if (!dst || !src) return 0;
    v41_row_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, g_cur_stream>>>(
        (float *)dst->ptr + dst_row * n, (const float *)src->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 row add");
}
