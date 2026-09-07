/* cuda_api_attention_4.inc.cu — 单 token 解码注意力分行核的发射(2026-09-05)。
 * 核在 cuda_attn_kernels_6.inc.cu; 两个解码入口(稠密 ds4_gpu_attention_decode_heads_tensor /
 * 稀疏 ds4_gpu_attention_indexed_mixed_batch_heads_tensor)在 n_tokens==1 时先试这里,
 * 返回 <0 = 暂存未就绪(捕获态里不能 cudaMalloc), 调用方退回老核。
 */
static float *g_attn_split_part = NULL;   /* [64 head][128 块][516] f32 = 17 MB */
static float *g_attn_split_part_b = NULL; /* 批版: [8 token][64 head][32 块][516] f32 = 34 MB */

/* 进 token 图捕获前预分配(cuda_decode_scratch_prepare 调用): 图内不能 cudaMalloc, 懒分配
 * 会让图路和直发路走不同核 —— 09-05 splitk 暂存就是这么分叉的。 */
static void cuda_attn_split_scratch_prepare(void) {
    if (!g_attn_split_part) {
        const size_t bytes = (size_t)64u * DS4_ATTN_SPLIT_MAXC * DS4_ATTN_SPLIT_PART * sizeof(float);
        if (cudaMalloc(&g_attn_split_part, bytes) != cudaSuccess) {
            g_attn_split_part = NULL; (void)cudaGetLastError();
        }
    }
    if (!g_attn_split_part_b) {
        const size_t bytes = (size_t)DS4_ATTN_SPLIT_BATCH_TOK * 64u * DS4_ATTN_SPLIT_MAXC_B * DS4_ATTN_SPLIT_PART * sizeof(float);
        if (cudaMalloc(&g_attn_split_part_b, bytes) != cudaSuccess) {
            g_attn_split_part_b = NULL; (void)cudaGetLastError();
        }
    }
}

static int attention_decode_split_launch(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const uint8_t *comp_kv,   /* 压缩缓存 f16 行 */
        const int32_t *topk,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t raw_first_idx,
        uint32_t raw_count,
        uint32_t visible_comp,
        uint32_t comp_count,
        uint32_t n_head) {
    if (n_head > 64u || (n_head & 7u) != 0u) return -1;
    if (!g_attn_split_part) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &cs);
        if (cs == cudaStreamCaptureStatusNone) cuda_attn_split_scratch_prepare();
        if (!g_attn_split_part) return -1;
    }
    const uint32_t n_score = raw_count + comp_count;
    uint32_t chunk_rows = DS4_ATTN_SPLIT_CHUNK;
    if (n_score > chunk_rows * DS4_ATTN_SPLIT_MAXC) {
        chunk_rows = (n_score + DS4_ATTN_SPLIT_MAXC - 1u) / DS4_ATTN_SPLIT_MAXC;
        chunk_rows = (chunk_rows + DS4_ATTN_SPLIT_STAGE - 1u) & ~(DS4_ATTN_SPLIT_STAGE - 1u);
    }
    uint32_t n_chunks = n_score ? (n_score + chunk_rows - 1u) / chunk_rows : 1u;
    if (n_chunks > DS4_ATTN_SPLIT_MAXC) return -1;
    dim3 grid(n_chunks, n_head / DS4_ATTN_SPLIT_HEADS, 1);
    ds4_launch_pdl(attention_decode_split_kernel, grid, 256, 0, g_cur_stream, 
            g_attn_split_part, q, raw_kv, comp_kv ? comp_kv : (const uint8_t *)raw_kv, topk,
            raw_cap, raw_start, raw_first_idx, raw_count, visible_comp, comp_count, chunk_rows, n_head);
    if (!cuda_ok(cudaGetLastError(), "attention decode split launch")) return 0;
    ds4_launch_pdl(attention_decode_merge_kernel, n_head, 256, 0, g_cur_stream,
            heads, g_attn_split_part, sinks, n_chunks, n_head);
    return cuda_ok(cudaGetLastError(), "attention decode merge launch");
}

/* 小批逐 token 分行核(2026-09-07, 投机 verify ≤8 token): 每 token 按单 token 解码同式算可见范围(原始窗 [lo,hi] 由
 * 批的线性化窗 first_raw_pos 起算, 压缩行 (qpos+1)/ratio 或 topk 行), 逐 token 发分行核 ⇒ 注意力与纯解码逐位同轨
 * (批核 attention_decode_mixed 的分块/归约序不同, 是投机输出分叉的根因之一)。返回 <0 = 不适用(调用方回退批核)。 */
static int attention_decode_split_tokens(float *heads, const float *sinks, const float *q, const float *raw_kv,
                                         const uint8_t *comp_kv, const int32_t *topk, uint32_t top_k,
                                         uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
                                         uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio,
                                         uint32_t n_head) {
    if (n_tokens > DS4_ATTN_SPLIT_BATCH_TOK || n_raw == 0u || n_raw > pos0 + n_tokens) return -1;
    if (n_head > 64u || (n_head & 7u) != 0u) return -1;
    if (!g_attn_split_part_b) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &cs);
        if (cs == cudaStreamCaptureStatusNone) cuda_attn_split_scratch_prepare();
        if (!g_attn_split_part_b) return -1;
    }
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
    ds4_attn_tokp p; memset(&p, 0, sizeof(p));
    uint32_t max_chunks = 0u, batch_ok = 1u;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t qpos = pos0 + t;
        uint32_t raw_count = 0u, raw_first_idx = 0u;
        if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window && qpos + 1u - window > lo) lo = qpos + 1u - window;
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) { raw_first_idx = lo - first_raw_pos; raw_count = hi - lo + 1u; }
            if (raw_count > 256u) raw_count = 256u;
        }
        uint32_t visible_comp = (ratio != 0u) ? (qpos + 1u) / ratio : n_comp;
        if (visible_comp > n_comp) visible_comp = n_comp;
        p.raw_first_idx[t] = raw_first_idx; p.raw_count[t] = raw_count;
        p.visible_comp[t] = visible_comp; p.comp_count[t] = topk ? top_k : visible_comp;
        /* 分块与单 token 发射(attention_decode_split_launch)逐字同式 ⇒ 每 token 的部分和与单 token 核逐位同;
         * 09-07 去掉 "行数 > CHUNK×MAXC 就逐 token 单发" 的封顶(1M ratio-128 层 8320 行), MAXC_B 已抬到 MAXC。 */
        const uint32_t n_score = raw_count + p.comp_count[t];
        uint32_t chunk_rows = DS4_ATTN_SPLIT_CHUNK;
        if (n_score > chunk_rows * DS4_ATTN_SPLIT_MAXC) {
            chunk_rows = (n_score + DS4_ATTN_SPLIT_MAXC - 1u) / DS4_ATTN_SPLIT_MAXC;
            chunk_rows = (chunk_rows + DS4_ATTN_SPLIT_STAGE - 1u) & ~(DS4_ATTN_SPLIT_STAGE - 1u);
        }
        p.chunk_rows[t] = chunk_rows;
        p.n_chunks[t] = n_score ? (n_score + chunk_rows - 1u) / chunk_rows : 1u;
        if (p.n_chunks[t] > DS4_ATTN_SPLIT_MAXC_B) batch_ok = 0u;
        if (p.n_chunks[t] > max_chunks) max_chunks = p.n_chunks[t];
    }
    if (batch_ok) {   /* 一发覆盖全部 token: grid.z = token 各扫各的行。
                       * (09-07 试过"一个 warp 同时给 4 个 query 算同一条 K 行"的多 query 核: 寄存器 ~170/线程压到每 SM 一块,
                       *  1M 剖面 36 ms/轮 反比本核 23 慢, 且短提示分叉 ⇒ 删。) */
        dim3 grid(max_chunks, n_head / DS4_ATTN_SPLIT_HEADS, n_tokens);
        ds4_launch_pdl(attention_decode_split_batch_kernel, grid, 256, 0, g_cur_stream,
                       g_attn_split_part_b, q, raw_kv, comp_kv ? comp_kv : (const uint8_t *)raw_kv, topk, top_k,
                       raw_cap, raw_start, p, n_head);
        if (!cuda_ok(cudaGetLastError(), "attention decode split batch launch")) return 0;
        ds4_launch_pdl(attention_decode_merge_batch_kernel, dim3(n_head, n_tokens, 1), 256, 0, g_cur_stream,
                       heads, g_attn_split_part_b, sinks, p, n_head);
        return cuda_ok(cudaGetLastError(), "attention decode merge batch launch");
    }
    for (uint32_t t = 0; t < n_tokens; t++) {   /* 长上下文(块数超批版暂存): 逐 token 单发, 数值同 */
        const uint64_t off = (uint64_t)t * n_head * 512u;
        const int r = attention_decode_split_launch(heads + off, sinks, q + off, raw_kv, n_comp ? comp_kv : NULL,
                                                    topk ? topk + (uint64_t)t * top_k : NULL, raw_cap, raw_start,
                                                    p.raw_first_idx[t], p.raw_count[t], p.visible_comp[t],
                                                    p.comp_count[t], n_head);
        if (r <= 0) return r;   /* 首 token 不适用 → -1 回退批核; 0 = 发射失败 */
    }
    return 1;
}
