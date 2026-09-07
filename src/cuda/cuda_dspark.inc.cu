/* cuda_dspark.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * DSpark drafter 专用 kernels 与 API。
 */
__global__ static void hc_mean_slot_kernel(
        float *dst, const float *hc, uint32_t n_embd, uint32_t n_hc,
        uint32_t slot, uint32_t n_tokens);

/* ================= DSpark drafter 专用 kernels (2026-08-18) =================
 * 语义对齐 hf/inference/model.py DSparkAttention/forward_head:
 *  - KV 窗: [win=128] 行来自主模型融合 main_x 的 wkv(逐 decode 步写 pos%win), 环形;
 *  - 块内: 5 位草稿自身 kv, 因果(位 i 看窗全部 + 块内 ≤i);
 *  - sink + softmax scale + 输出反 rope 由调用方现有 entry 完成;
 *  - drafter KV 全 f32(草稿路径, verify 兜底正确性, 不追官方 fp8 bit 对齐)。 */
/* prefill 建窗 scatter: [n,512] 行写入环形窗 (pos0+t)%win */
__global__ static void dspark_win_scatter_kernel(
        float *win, const float *rows, uint32_t n, uint32_t pos0,
        uint32_t win_rows, uint32_t dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= n || i >= dim) return;
    win[(uint64_t)((pos0 + t) % win_rows) * dim + i] = rows[(uint64_t)t * dim + i];
}

__global__ static void dspark_attn_kernel(
        float *heads,                  /* [blk, n_head, head_dim] */
        const float *sinks,            /* [n_head] */
        const float *q,                /* [blk, n_head, head_dim] (已 rope) */
        const float *win_kv,           /* [win, head_dim] 环形(行序=写入序) */
        const float *blk_kv,           /* [blk, head_dim] (已 rope) */
        uint32_t n_win,                /* 窗内有效行数(≤win) */
        uint32_t blk,                  /* 块位数(5) */
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t win_base,             /* 窗首位置在环里的行号 = (pos+1-n_win) % win_cap */
        uint32_t win_cap) {            /* 环容量(128) */
    const uint32_t t = blockIdx.x;     /* 块位 */
    const uint32_t h = blockIdx.y;     /* head */
    if (t >= blk || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const uint32_t n_score = n_win + blk;      /* 窗 + 块内全可见(官方 topk_idxs:
                                                * 每块位看到全部 5 位, 非因果; 块内
                                                * 因果性只在 markov 链的 logits 层面) */
    __shared__ float scores[256];
    __shared__ float partial[32];
    __shared__ float max_s, denom;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t wp = threadIdx.x >> 5u, ln = threadIdx.x & 31u;
    const uint32_t hd4 = head_dim >> 2u;
    const float4 *q4 = (const float4 *)qh;
    for (uint32_t r = wp; r < n_score; r += blockDim.x >> 5u) {
        /* 环行按**位置**映射(2026-08-21 修): 原来读 ring[0..n_win-1] —— 上下文过 128 后
         * 会把 verify 批写进去的"被拒候选(未来位置)"行一起读进来污染草稿(实测 SPEC 下
         * 等效 p1 只有 0.62, 而 PROBE 干净窗口下 drafter 真实 p1=0.91)。 */
        const uint32_t ring = (win_cap && r < n_win) ? ((win_base + r) % win_cap) : r;
        const float *kvrow = (r < n_win) ? (win_kv + (uint64_t)ring * head_dim)
                                         : (blk_kv + (uint64_t)(r - n_win) * head_dim);
        const float4 *kv4 = (const float4 *)kvrow;
        float dot = 0.0f;
        for (uint32_t d = ln; d < hd4; d += 32u) {
            const float4 k = kv4[d], qq = q4[d];
            dot += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
        }
        for (uint32_t off = 16u; off > 0u; off >>= 1u) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (ln == 0) scores[r] = dot * scale;
    }
    __syncthreads();
    float lm = sinks[h];
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) lm = fmaxf(lm, scores[i]);
    for (uint32_t off = 16u; off > 0u; off >>= 1u) lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, off));
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = lm;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float m = (threadIdx.x < nw) ? partial[threadIdx.x] : -INFINITY;
        for (uint32_t off = 16u; off > 0u; off >>= 1u) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) max_s = m;
    }
    __syncthreads();
    float dl = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        dl += scores[i];
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u) dl += __shfl_down_sync(0xffffffffu, dl, off);
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = dl;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float s2 = (threadIdx.x < nw) ? partial[threadIdx.x] : 0.0f;
        for (uint32_t off = 16u; off > 0u; off >>= 1u) s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if (threadIdx.x == 0) denom = s2 + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_score; r++) {
            const uint32_t ring2 = (win_cap && r < n_win) ? ((win_base + r) % win_cap) : r;
            const float *kvrow = (r < n_win) ? (win_kv + (uint64_t)ring2 * head_dim)
                                             : (blk_kv + (uint64_t)(r - n_win) * head_dim);
            acc += kvrow[d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

/* DSpark 置信头(2026-08-21, 论文 2607.05147 Alg.1 的输入量):
 *   c_k = sigmoid( w · [ x_k ; W1[prev_k] ] ),  w 是 [dim+rank] f32, W1 是 q8_0 [vocab, rank]。
 * x_k = hc_head 之后、norm 之前的 drafter 隐状态(与官方 model.py DSparkConfidenceHead 一致)。
 * 一次算完 block 内全部位置: 每块一个位置, 块内并行点积 + 固定序归约。 */
__global__ static void dspark_confidence_kernel(
        float *out_conf,               /* [n_pos] */
        const float *x,                /* [n_pos, dim] hc_head 输出(pre-norm) */
        const float *w,                /* [dim + rank] f32 */
        const char *w1_q8,             /* q8_0 [vocab, rank] */
        const int32_t *prev_ids,       /* [n_pos] 每位置的输入 token */
        uint32_t dim, uint32_t rank, uint32_t n_pos) {
    const uint32_t p = blockIdx.x;
    if (p >= n_pos) return;
    const float *xp = x + (uint64_t)p * dim;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) acc += w[i] * xp[i];
    const uint32_t rank_blocks = rank / 32u;
    const int32_t prev = prev_ids[p];
    for (uint32_t b = threadIdx.x; b < rank_blocks; b += blockDim.x) {
        const uint8_t *blkp = (const uint8_t *)(w1_q8 + ((uint64_t)prev * rank_blocks + b) * 34u);
        uint16_t dh; memcpy(&dh, blkp, 2);
        const float d = __half2float(__ushort_as_half(dh));
        const int8_t *qs = (const int8_t *)(blkp + 2);
        float s = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) s += (float)qs[i] * w[dim + b * 32u + i];
        acc += d * s;
    }
    __shared__ float part[256];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) part[threadIdx.x] += part[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out_conf[p] = 1.0f / (1.0f + __expf(-part[0]));
}

int ds4_gpu_dspark_confidence_tensor(
        ds4_gpu_tensor *out_conf,
        const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t conf_w_offset, uint64_t markov_w1_offset,
        const ds4_gpu_tensor *prev_ids,
        uint32_t dim, uint32_t rank, uint32_t vocab, uint32_t n_pos) {
    if (!out_conf || !x || !prev_ids || !model_map || n_pos == 0 || (rank % 32u) != 0u) return 0;
    const uint64_t wb = (uint64_t)(dim + rank) * sizeof(float);
    const uint64_t w1b = (uint64_t)vocab * (rank / 32u) * 34ull;
    const char *w = cuda_model_range_ptr(model_map, conf_w_offset, wb, "dspark_conf");
    const char *w1 = cuda_model_range_ptr(model_map, markov_w1_offset, w1b, "dspark_markov_w1");
    if (!w || !w1) return 0;
    dspark_confidence_kernel<<<n_pos, 256, 0, g_cur_stream>>>(
            (float *)out_conf->ptr, (const float *)x->ptr, (const float *)w, w1,
            (const int32_t *)prev_ids->ptr, dim, rank, n_pos);
    return cuda_ok(cudaGetLastError(), "dspark confidence");
}

/* markov 链: 每位置 logits += w2 @ w1[prev]; argmax 选出下一位 prev(串行 5 次调用,
 * 每次一个位置)。w1: [rank=256] 行 per token(q8_0 反量化在此内联? 简化: w1/w2 由调用
 * 方保证 f32 缓存(2×129280×256×4B=265MB 太大) —— 改: w1 行(256 f32)由小 kernel 从
 * q8_0 现场解一行, w2 是 [vocab,256] q8_0 逐行 dot。 */
__global__ static void dspark_markov_bias_argmax_kernel(
        int32_t *out_id,               /* [1] 本位置 argmax */
        float *logits,                 /* [vocab] 本位置(原地加 bias) */
        const char *w1_q8,             /* q8_0 [vocab, 256] */
        const char *w2_q8,             /* q8_0 [vocab, 256] */
        const int32_t *prev_id,        /* [1] */
        uint32_t vocab,
        uint32_t rank) {
    /* 阶段1: 块 0 解 w1[prev] 到共享 → 全网格用 __grid_constant__ 不行, 用两 kernel?
     * 简化: 每块自解 w1[prev](256 元素 q8_0 = 8 块×34B, 开销小), 各块算自己 vocab 段的
     * bias 并加到 logits; argmax 用原子(int 打包 value|idx)。 */
    __shared__ float w1row[256];
    const uint32_t rank_blocks = rank / 32u;
    const int32_t prev = *prev_id;
    for (uint32_t b = threadIdx.x; b < rank_blocks; b += blockDim.x) {
        const uint8_t *blkp = (const uint8_t *)(w1_q8 + ((uint64_t)prev * rank_blocks + b) * 34u);
        uint16_t dh; memcpy(&dh, blkp, 2);
        const float d = __half2float(__ushort_as_half(dh));
        const int8_t *qs = (const int8_t *)(blkp + 2);
        for (uint32_t i = 0; i < 32u; i++) w1row[b * 32u + i] = d * (float)qs[i];
    }
    __syncthreads();
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < vocab) {
        const uint8_t *row = (const uint8_t *)(w2_q8 + (uint64_t)v * rank_blocks * 34u);
        float bias = 0.0f;
        for (uint32_t b = 0; b < rank_blocks; b++) {
            const uint8_t *blkp = row + b * 34u;
            uint16_t dh; memcpy(&dh, blkp, 2);
            const float d = __half2float(__ushort_as_half(dh));
            const int8_t *qs = (const int8_t *)(blkp + 2);
            float s = 0.0f;
            for (uint32_t i = 0; i < 32u; i++) s += (float)qs[i] * w1row[b * 32u + i];
            bias += d * s;
        }
        logits[v] += bias;
    }
    /* argmax 二段: 由独立 kernel 做(见 dspark_argmax_kernel) */
    (void)out_id;
}

__global__ static void dspark_argmax_kernel(int32_t *out_id, const float *logits, uint32_t vocab) {
    __shared__ float bm[256];
    __shared__ int32_t bi[256];
    float m = -INFINITY; int32_t mi = 0;
    for (uint32_t v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float x = logits[v];
        if (x > m) { m = x; mi = (int32_t)v; }
    }
    bm[threadIdx.x] = m; bi[threadIdx.x] = mi;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (bm[threadIdx.x + s] > bm[threadIdx.x] ||
                (bm[threadIdx.x + s] == bm[threadIdx.x] && bi[threadIdx.x + s] < bi[threadIdx.x])) {
                bm[threadIdx.x] = bm[threadIdx.x + s]; bi[threadIdx.x] = bi[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out_id = bi[0];
}

/* DSpark: 抓 target 层 HC 均值到 main_hidden[t][slot]; n_tokens 支持 batch prefill */
int ds4_gpu_dspark_hc_mean_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *hc,
        uint32_t n_embd, uint32_t n_hc, uint32_t slot, uint32_t n_tokens) {
    if (!dst || !hc || n_embd == 0 || n_hc == 0 || slot >= 3u || n_tokens == 0) return 0;
    if (hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float) ||
        dst->bytes < (uint64_t)n_tokens * 3u * n_embd * sizeof(float)) return 0;
    hc_mean_slot_kernel<<<dim3((n_embd + 255u) / 256u, n_tokens, 1), 256, 0, g_cur_stream>>>(
        (float *)dst->ptr, (const float *)hc->ptr, n_embd, n_hc, slot, n_tokens);
    return cuda_ok(cudaGetLastError(), "dspark hc mean launch");
}

int ds4_gpu_dspark_attn_tensor(
        ds4_gpu_tensor *heads,
        const void *model_map, uint64_t model_size, uint64_t sinks_offset,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *win_kv, const ds4_gpu_tensor *blk_kv,
        uint32_t n_win, uint32_t blk, uint32_t n_head, uint32_t head_dim, uint32_t win_base, uint32_t win_cap) {
    if (!heads || !q || !win_kv || !blk_kv || blk == 0 || n_head == 0 || head_dim == 0) return 0;
    if ((head_dim & 3u) != 0 || n_win + blk > 256u) return 0;
    if (sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) return 0;
    const char *sinks = cuda_model_range_ptr(model_map, sinks_offset,
                                             (uint64_t)n_head * sizeof(float), "dspark_sinks");
    if (!sinks) return 0;
    dspark_attn_kernel<<<dim3(blk, n_head, 1), 256, 0, g_cur_stream>>>(
        (float *)heads->ptr, (const float *)sinks, (const float *)q->ptr,
        (const float *)win_kv->ptr, (const float *)blk_kv->ptr,
        n_win, blk, n_head, head_dim, win_base, win_cap);
    return cuda_ok(cudaGetLastError(), "dspark attn launch");
}

int ds4_gpu_dspark_win_scatter_tensor(
        ds4_gpu_tensor *win, const ds4_gpu_tensor *rows,
        uint32_t n, uint32_t pos0, uint32_t win_rows, uint32_t dim) {
    if (!win || !rows || n == 0 || win_rows == 0 || dim == 0) return 0;
    if (rows->bytes < (uint64_t)n * dim * sizeof(float) ||
        win->bytes < (uint64_t)win_rows * dim * sizeof(float)) return 0;
    dspark_win_scatter_kernel<<<dim3(n, (dim + 255u) / 256u, 1), 256, 0, g_cur_stream>>>(
        (float *)win->ptr, (const float *)rows->ptr, n, pos0, win_rows, dim);
    return cuda_ok(cudaGetLastError(), "dspark win scatter launch");
}


/* markov 一步: logits 行原地加 bias(w2@w1[prev]) 并 argmax → out_id(device int32) */
int ds4_gpu_dspark_markov_step_tensor(
        ds4_gpu_tensor *out_id, ds4_gpu_tensor *logits_row,
        const void *model_map, uint64_t model_size,
        uint64_t w1_offset, uint64_t w2_offset,
        const ds4_gpu_tensor *prev_id,
        uint32_t vocab, uint32_t rank) {
    if (!out_id || !logits_row || !prev_id || vocab == 0 || rank == 0 || (rank & 31u)) return 0;
    const uint64_t wbytes = (uint64_t)vocab * (rank / 32u) * 34u;
    if (w1_offset > model_size || wbytes > model_size - w1_offset ||
        w2_offset > model_size || wbytes > model_size - w2_offset) return 0;
    const char *w1 = cuda_model_range_ptr(model_map, w1_offset, wbytes, "dspark_markov_w1");
    const char *w2 = cuda_model_range_ptr(model_map, w2_offset, wbytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;
    dspark_markov_bias_argmax_kernel<<<(vocab + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (int32_t *)out_id->ptr, (float *)logits_row->ptr, w1, w2,
        (const int32_t *)prev_id->ptr, vocab, rank);
    dspark_argmax_kernel<<<1, 256, 0, g_cur_stream>>>(
        (int32_t *)out_id->ptr, (const float *)logits_row->ptr, vocab);
    return cuda_ok(cudaGetLastError(), "dspark markov launch");
}

int ds4_gpu_dspark_argmax_only_tensor(ds4_gpu_tensor *out_id,
                                                 const ds4_gpu_tensor *logits_row, uint32_t vocab) {
    if (!out_id || !logits_row || vocab == 0) return 0;
    dspark_argmax_kernel<<<1, 256, 0, g_cur_stream>>>(
        (int32_t *)out_id->ptr, (const float *)logits_row->ptr, vocab);
    return cuda_ok(cudaGetLastError(), "dspark argmax launch");
}

static void tok_graph_ensure_id_slot(void) {
    side_stream_ensure();
    if (!g_tok_id_dev &&
        cudaMalloc(&g_tok_id_dev, sizeof(int32_t)) != cudaSuccess) {
        g_tok_id_dev = NULL; (void)cudaGetLastError();
    }
    if (!g_tok_id_host &&
        cudaMallocHost(&g_tok_id_host, sizeof(int32_t)) != cudaSuccess) {
        g_tok_id_host = NULL; (void)cudaGetLastError();
    }
    /* 预发射资源: 四相 {mode,id} pinned 槽(设备零拷贝读)、设备 argmax 槽、回传事件 */
    if (!g_tok_slots_host) {
        if (cudaHostAlloc(&g_tok_slots_host, 8u * sizeof(int32_t), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&g_tok_slots_dev, g_tok_slots_host, 0) != cudaSuccess) {
            g_tok_slots_host = NULL; g_tok_slots_dev = NULL; (void)cudaGetLastError();
        } else {
            memset(g_tok_slots_host, 0, 8u * sizeof(int32_t));
        }
    }
    if (!g_tok_next_dev && cudaMalloc(&g_tok_next_dev, sizeof(int32_t)) != cudaSuccess) {
        g_tok_next_dev = NULL; (void)cudaGetLastError();
    }
    if (!g_argmax_pm && cudaMalloc(&g_argmax_pm, 64u * (sizeof(float) + sizeof(int32_t))) != cudaSuccess) {
        g_argmax_pm = NULL; (void)cudaGetLastError();
    }
    if (g_argmax_pm) g_argmax_pi = (int32_t *)(g_argmax_pm + 64);
    if (!g_readback_ev && cudaEventCreateWithFlags(&g_readback_ev, cudaEventDisableTiming) != cudaSuccess) {
        g_readback_ev = NULL; (void)cudaGetLastError();
    }
}

/* graph → exec 槽: 首次 Instantiate, 之后 ExecUpdate, 不匹配则重建 */
static int tok_graph_finalize_slot(cudaGraph_t graph, int slot) {
    if (!g_tok_execs[slot]) {
        if (cudaGraphInstantiate(&g_tok_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            g_tok_execs[slot] = NULL;
            return 0;
        }
        return 1;
    }
    cudaGraphExecUpdateResult ur;
    cudaGraphNode_t err_node = NULL;
    if (cudaGraphExecUpdate(g_tok_execs[slot], graph, &err_node, &ur) != cudaSuccess) {
        /* 910 竞争根修(08-18 sanitizer 实锤)修订(08-20): 原修=全设备同步再销毁, 但该同步
         * 等的是"刚发射的当前图"(27.5ms 隔拍阻塞实测) — 而被销毁的 exec 在两处调用点都
         * 满足"上次发射 ≥1 个 end_commands 全同步之前"不变量(每 token 末尾必 DeviceSync),
         * 恒已静默。撤同步; DS4_TOK_GRAPH_SAFE_SYNC=1 可恢复旧行为(sanitizer 排查用)。 */
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] update REJECT ur=%d\n", (int)ur);
        (void)cudaGetLastError();
        if (((const char *)0) /* DS4_TOK_GRAPH_SAFE_SYNC: 路径开关已删(2026-08-22 隐形炸弹清理) */) (void)cudaDeviceSynchronize();
        (void)cudaGraphExecDestroy(g_tok_execs[slot]);
        g_tok_execs[slot] = NULL;
        if (cudaGraphInstantiate(&g_tok_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
    }
    return 1;
}

