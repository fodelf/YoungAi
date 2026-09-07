/* cuda_attn_kernels_6.inc.cu — 单 token 解码注意力"分行在线 + 合并"核(2026-09-05)。
 *
 * 为什么: 老解码核(attention_decode_mixed / attention_indexed_mixed)每个 head 一个 block
 * (grid 1×64), 一个 block 串行扫 ≤768 行且每行读两遍(打分一遍、加权和一遍); n_head_kv=1
 * 意味着 64 个 head 把同一份 KV 从 L2 重读 64 次 ⇒ 200 MB/层, 稀疏层实测 346 µs/层,
 * 4096 ctx 一 token 光注意力吃 7.1 ms(45→56 ms/token, bench 20→16 t/s 的主因)。
 * 这里 block = 8 head × 一个行块: 每行进 shared 一次供 8 个 head 用(L2 流量降 8×), 行块跨
 * block 并行(grid = 行块数 × 8 组, 48 SM 全占), 每 head 在线 softmax 出 (o, m, l) 部分和,
 * 再由合并核按 head 归一。数学与 prefill 批路现役的 attention_indexed_mixed_heads8_online
 * 同式; 与老两遍核只差浮点序(容差级), 判决走 kernel_parity(FP 锚五指标)。
 * 改了会怎样: 行块分得太细 ⇒ 部分和写读放大(每块 2 KB×64 head); 太粗 ⇒ block 数不够
 * 填 SM。DS4_ATTN_SPLIT_CHUNK=64 行: 稀疏 640 行 → 10×8=80 block, 稠密上限 1152 行 → 18×8。
 */
#define DS4_ATTN_SPLIT_HEADS 8u
#define DS4_ATTN_SPLIT_STAGE 8u
/* 09-06 MAXC 32→128: 1M 上下文 ratio-128 层解码要扫 ~9216 行, 32 块封顶 ⇒ 每块 288 行串行、每层只 256 个 block,
 * 分行核 155 µs/层(6.7 ms/token); 128 块 ⇒ 72 行/块、1024 block。短上下文(≤ 32×32 行)块数不变。暂存 64×128×516×4 = 17 MB。 */
#define DS4_ATTN_SPLIT_MAXC  128u
/* 09-06 K4: 64→32 行/块。2048 ctx 稀疏层 n_score=640 ⇒ 10 块×8 组=80 block 填不满 48 SM×多驻留
 * (33 µs/层); 32 行 ⇒ 20×8=160 block。部分和写读量翻倍(每块 2 KB×64 head)仍远小于 KV 读量。 */
#define DS4_ATTN_SPLIT_CHUNK 32u
#define DS4_ATTN_SPLIT_PART  516u   /* o[512] + m + l + 2 pad: 行距 2064 B 保 float4 对齐 */

/* 核体抽成 device 函数(09-07): 单 token 核与批版(grid.z = token, 投机 verify 批)共用同一份 ⇒ 每 token 逐位同。
 * (chunk, head) 由调用核给出; part/q/topk 已按 token 平移。 */
__device__ static void attn_split_body(
        float *part,               /* [n_head][MAXC][PART] */
        const float *q,            /* [n_head][512] */
        const float *raw_kv,
        const uint8_t *comp_kv,     /* 压缩缓存 f16 行 */
        const int32_t *topk,       /* NULL ⇒ 压缩行取 [0, comp_count) 稠密; 否则 topk[i], 越界/负=跳过 */
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t raw_first_idx,
        uint32_t raw_count,
        uint32_t visible_comp,
        uint32_t comp_count,
        uint32_t chunk_rows,
        uint32_t n_head,
        uint32_t chunk,
        uint32_t head) {
    const uint32_t lane = threadIdx.x & 31u;
    const bool valid_head = head < n_head;
    const uint32_t n_score = raw_count + comp_count;
    const uint32_t row_lo = chunk * chunk_rows;
    const uint32_t row_hi = row_lo + chunk_rows < n_score ? row_lo + chunk_rows : n_score;

    __shared__ float4 kv_shared[DS4_ATTN_SPLIT_STAGE * 128];
    __shared__ const void *row_ptr[DS4_ATTN_SPLIT_STAGE];    /* NULL = 无效行(topk 越界) */
    __shared__ uint32_t row_is_comp[DS4_ATTN_SPLIT_STAGE];   /* 1 = 压缩缓存 f16 行, 0 = 原始 f32 行 */

    const float scale = rsqrtf(512.0f);
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        const float4 *q4 = (const float4 *)(q + (uint64_t)head * 512u);
        q0 = q4[lane]; q1 = q4[lane + 32u]; q2 = q4[lane + 64u]; q3 = q4[lane + 96u];
    }
    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = row_lo; row0 < row_hi; row0 += DS4_ATTN_SPLIT_STAGE) {
        const uint32_t nr = row_hi - row0 < DS4_ATTN_SPLIT_STAGE ? row_hi - row0 : DS4_ATTN_SPLIT_STAGE;
        if (threadIdx.x < nr) {
            const uint32_t sr = row0 + threadIdx.x;
            const void *p = NULL;
            uint32_t is_comp = 0u;
            if (sr < raw_count) {
                p = raw_kv + (uint64_t)((raw_start + raw_first_idx + sr) % raw_cap) * 512u;
            } else {
                const uint32_t ci = sr - raw_count;
                is_comp = 1u;
                if (topk) {
                    const int32_t c = topk[ci];
                    if (c >= 0 && (uint32_t)c < visible_comp) p = COMP_ROW(comp_kv, c);
                } else if (ci < visible_comp) {
                    p = COMP_ROW(comp_kv, ci);
                }
            }
            row_ptr[threadIdx.x] = p;
            row_is_comp[threadIdx.x] = is_comp;
        }
        __syncthreads();
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const void *p = row_ptr[off >> 7u];
            if (p) kv_shared[off] = row_is_comp[off >> 7u] ? ld_comp4((const uint8_t *)p, off & 127u)
                                                           : ((const float4 *)p)[off & 127u];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                if (!row_ptr[rr]) continue;
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane], k1 = kv4[lane + 32u], k2 = kv4[lane + 64u], k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) + dot4_f32(q1, k1) + dot4_f32(q2, k2) + dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale; o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale; o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale; o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale; o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale; o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale; o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale; o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale; o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float *ph = part + ((uint64_t)head * DS4_ATTN_SPLIT_MAXC + chunk) * DS4_ATTN_SPLIT_PART;
        float4 *o4 = (float4 *)ph;
        o4[lane] = o0; o4[lane + 32u] = o1; o4[lane + 64u] = o2; o4[lane + 96u] = o3;
        if (lane == 0) { ph[512] = max_s; ph[513] = sum_s; }
    }
}
__global__ static void attention_decode_split_kernel(
        float *part, const float *q, const float *raw_kv, const uint8_t *comp_kv, const int32_t *topk,
        uint32_t raw_cap, uint32_t raw_start, uint32_t raw_first_idx, uint32_t raw_count,
        uint32_t visible_comp, uint32_t comp_count, uint32_t chunk_rows, uint32_t n_head) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    attn_split_body(part, q, raw_kv, comp_kv, topk, raw_cap, raw_start, raw_first_idx, raw_count, visible_comp,
                    comp_count, chunk_rows, n_head, blockIdx.x, blockIdx.y * DS4_ATTN_SPLIT_HEADS + (threadIdx.x >> 5u));
}
/* 批版(投机 verify ≤8 token): grid.z = token, 每 token 自己的可见范围/分块(按单 token 同式在 host 算好, 按值传),
 * part 按 token 平移 DS4_ATTN_SPLIT_MAXC_B 块; chunk ≥ 本 token 块数的 block 直接退出。
 * 09-07 MAXC_B 32 → 128(= 单 token 的 MAXC): 1M 上下文 ratio-128 层一 token 扫 8320 行 = 116 块, 32 块封顶时 4 个 token
 * 只能逐个单发(剖面 23 ms/轮 vs 一发 ~7), 而 4 token 的 block 数(116×8×4)本就能填满 48 SM。暂存 8 tok×64×128×2064 B = 135 MB。
 * chunk_rows 逐 token 按单 token 公式算(行数 > 32×128 时按 128 块均分并对齐 STAGE) ⇒ 分块与单 token 核逐位同。 */
#define DS4_ATTN_SPLIT_MAXC_B 128u
#define DS4_ATTN_SPLIT_BATCH_TOK 8u
typedef struct {
    uint32_t raw_first_idx[DS4_ATTN_SPLIT_BATCH_TOK], raw_count[DS4_ATTN_SPLIT_BATCH_TOK];
    uint32_t visible_comp[DS4_ATTN_SPLIT_BATCH_TOK], comp_count[DS4_ATTN_SPLIT_BATCH_TOK];
    uint32_t n_chunks[DS4_ATTN_SPLIT_BATCH_TOK], chunk_rows[DS4_ATTN_SPLIT_BATCH_TOK];
} ds4_attn_tokp;
__global__ static void attention_decode_split_batch_kernel(
        float *part, const float *q, const float *raw_kv, const uint8_t *comp_kv, const int32_t *topk,
        uint32_t top_k, uint32_t raw_cap, uint32_t raw_start, ds4_attn_tokp p, uint32_t n_head) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t t = blockIdx.z, chunk = blockIdx.x;
    if (chunk >= p.n_chunks[t]) return;
    /* 单 token 核的 part 行距是 MAXC 块, 这里每 token 留 MAXC_B 块: 用 head 步长换算 —— 把 part 视作
     * [tok][head][MAXC_B][PART], 传给核体时按 head 平移让它按 MAXC 寻址会越界, 所以核体拿到的是 "head 0 且 MAXC 视角"
     * 的假基址: base + (t*n_head + head)*MAXC_B*PART - head*MAXC*PART。 */
    const uint32_t head = blockIdx.y * DS4_ATTN_SPLIT_HEADS + (threadIdx.x >> 5u);
    float *pbase = part + ((uint64_t)t * n_head + head) * DS4_ATTN_SPLIT_MAXC_B * DS4_ATTN_SPLIT_PART
                        - (uint64_t)head * DS4_ATTN_SPLIT_MAXC * DS4_ATTN_SPLIT_PART;
    attn_split_body(pbase, q + (uint64_t)t * n_head * 512u, raw_kv, comp_kv,
                    topk ? topk + (uint64_t)t * top_k : NULL, raw_cap, raw_start,
                    p.raw_first_idx[t], p.raw_count[t], p.visible_comp[t], p.comp_count[t], p.chunk_rows[t], n_head, chunk, head);
}

/* 按 head 合并行块部分和: M = max(sink, m_c); L = e^{sink-M} + Σ l_c e^{m_c-M}; o = Σ o_c e^{m_c-M} / L。
 * 空行块 m_c=-inf ⇒ 权 0, 天然跳过。 */
__device__ static void attn_merge_body(float *heads, const float *ph, const float *sinks, uint32_t n_chunks, uint32_t head) {
    /* (09-07 试过 m/l 进 shared + 权重只算一次 + load 四块展开: 224 → 224 µs 无收益 —— 本核是读部分和的带宽墙,
     *  1M 时 116 块 × 64 头 × 4 token × 2 KB = 59 MB/发 ≈ 245 µs@240 GB/s; 块数由单 token 同分块逐位同的要求钉死。) */
    const float sink = sinks[head];
    float M = sink;
    for (uint32_t c = 0; c < n_chunks; c++) M = fmaxf(M, ph[c * DS4_ATTN_SPLIT_PART + 512u]);
    float L = expf(sink - M);
    for (uint32_t c = 0; c < n_chunks; c++)
        L += ph[c * DS4_ATTN_SPLIT_PART + 513u] * expf(ph[c * DS4_ATTN_SPLIT_PART + 512u] - M);
    const uint32_t d0 = threadIdx.x, d1 = threadIdx.x + 256u;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint32_t c = 0; c < n_chunks; c++) {
        const float *pc = ph + c * DS4_ATTN_SPLIT_PART;
        const float w = expf(pc[512] - M);
        a0 += pc[d0] * w;
        a1 += pc[d1] * w;
    }
    const float inv = L == 0.0f ? 0.0f : 1.0f / L;
    heads[(uint64_t)head * 512u + d0] = a0 * inv;
    heads[(uint64_t)head * 512u + d1] = a1 * inv;
}
__global__ static void attention_decode_merge_kernel(float *heads, const float *part, const float *sinks,
                                                     uint32_t n_chunks, uint32_t n_head) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t head = blockIdx.x;
    if (head >= n_head) return;
    attn_merge_body(heads, part + (uint64_t)head * DS4_ATTN_SPLIT_MAXC * DS4_ATTN_SPLIT_PART, sinks, n_chunks, head);
}
__global__ static void attention_decode_merge_batch_kernel(float *heads, const float *part, const float *sinks,
                                                           ds4_attn_tokp p, uint32_t n_head) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    const uint32_t head = blockIdx.x, t = blockIdx.y;
    if (head >= n_head) return;
    attn_merge_body(heads + (uint64_t)t * n_head * 512u,
                    part + ((uint64_t)t * n_head + head) * DS4_ATTN_SPLIT_MAXC_B * DS4_ATTN_SPLIT_PART,
                    sinks, p.n_chunks[t], head);
}
