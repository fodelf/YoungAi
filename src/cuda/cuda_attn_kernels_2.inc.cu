/* cuda_attn_kernels_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * attention kernel 族(indexed_mixed/heads8_online/hc_split_weighted_sum_norm_fused/compressor_set_rows)。
 */
__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const uint8_t *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    /* 批/单同路(2026-08-21 verify 等价): 原条件把"有压缩槽且 n_tokens>1"甩到下面
     * 8-lane-per-row 分支, head_dim 上的求和序与单 token 的 warp-per-row 不同 ⇒ 差 1 ULP,
     * 再被下游 q8 激活量化(1 档=1/127)放大到 1e-4, 逐层滚成 logit 级偏差, 吃掉投机接受率。
     * 形状够就一律走 float4 warp-per-row: 逐位与 decode 一致。 */
    if ((head_dim & 3u) == 0u) {
        /* 重写(2026-08-17): 原 thread-per-row 标量版 = 每线程串行读整条 2KB 行,
         * warp 32 线程同时戳 32 个不同行完全不合并。现 warp-per-row + float4:
         * 32 lane 连续 16B 粒度合并读一行, shuffle 树归约。 */
        const uint32_t wp = threadIdx.x >> 5u, ln = threadIdx.x & 31u;
        const uint32_t hd4 = head_dim >> 2u;
        const float4 *q4 = (const float4 *)qh;
        for (uint32_t r = wp; r < raw_count; r += 8u) {
            const float4 *kv4 = (const float4 *)(raw_kv + (uint64_t)raw_rows[r] * head_dim);
            float dot = 0.0f;
            for (uint32_t d = ln; d < hd4; d += 32u) {
                const float4 k = kv4[d], qq = q4[d];
                dot += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
            }
            for (uint32_t off = 16u; off > 0u; off >>= 1u) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (ln == 0) scores[r] = dot * scale;
        }
        for (uint32_t c = wp; c < visible_comp; c += 8u) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const uint8_t *kvh = COMP_ROW(comp_kv, c);
                float dot = 0.0f;
                for (uint32_t d = ln; d < hd4; d += 32u) {
                    const float4 k = ld_comp4(kvh, d), qq = q4[d];
                    dot += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
                }
                for (uint32_t off = 16u; off > 0u; off >>= 1u) dot += __shfl_down_sync(0xffffffffu, dot, off);
                s = dot * scale + add;
            }
            if (ln == 0) scores[raw_count + c] = s;
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x)
            local_max = fmaxf(local_max, scores[i]);
    } else if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const uint8_t *kvrow = COMP_ROW(comp_kv, c);
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * ld_comp1(kvrow, d);
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *rawrow = NULL;
                const uint8_t *comprow = NULL;
                if (row < raw_count) {
                    rawrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) comprow = COMP_ROW(comp_kv, c);
                }
                float s = -INFINITY;
                if (rawrow || comprow) {
                    float dot = 0.0f;
                    if (rawrow) {
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * rawrow[d];
                    } else {
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * ld_comp1(comprow, d);
                    }
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    /* shuffle 规约版 max/denom(树形固定 ⇒ 确定): 原 8 轮 __syncthreads×2 */
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = local_max;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float m = (threadIdx.x < nw) ? partial[threadIdx.x] : -INFINITY;
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) max_s = m;
    }
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        den_local += __shfl_down_sync(0xffffffffu, den_local, off);
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = den_local;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float s2 = (threadIdx.x < nw) ? partial[threadIdx.x] : 0.0f;
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if (threadIdx.x == 0) denom = s2 + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const uint8_t *kv = COMP_ROW(comp_kv, c);
            acc0 += ld_comp1(kv, d0) * s;
            acc1 += ld_comp1(kv, d1) * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += ld_comp1(COMP_ROW(comp_kv, c), d) * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const uint8_t *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    /* 单线程保序压紧: atomicAdd 抢槽的槽序随调度漂移 ⇒ 后面 score/softmax 的
     * 浮点求和顺序每 run 不同, 温 0 输出在 ULP tie 处翻 token。indexer 的 topk
     * 顺序本身是确定的, 按它保序填充即可复现。 */
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp && comp_count < 512u)
                comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (comp_count > 512u) comp_count = 512u;
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float dot = 0.0f;
                if (row < raw_count) {
                    const float *kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                } else {
                    const uint8_t *kvrow = COMP_ROW(comp_kv, comp_rows[row - raw_count]);
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * ld_comp1(kvrow, d);
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const uint8_t *kv = COMP_ROW(comp_kv, comp_rows[c]);
            acc0 += ld_comp1(kv, d0) * s;
            acc1 += ld_comp1(kv, d1) * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += ld_comp1(COMP_ROW(comp_kv, comp_rows[s]), d) * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

