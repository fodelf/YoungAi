/* cuda_api_embed_indexer_1.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * embed/indexer/argmax/topk_mask GPU API。
 */
int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    if (g_tok_id_dev && g_tok_slots_host && g_tok_next_dev) {
        /* 间接路径(09-07 改为核取槽, 见 cuda_lifecycle g_tok_slots_host 注): capture 期只记住相位
         * (图重放时槽里是发射前写的 {mode,id}); 直跑时立即写槽即刻消费(同线程同流串行, 安全)。 */
        cudaStreamCaptureStatus ecs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(cudaStreamPerThread, &ecs);
        g_tok_id_want = (int32_t)token;
        const int slot = (int)(g_tok_launch_pos & 3u);
        if (ecs == cudaStreamCaptureStatusNone) tok_slot_write(slot, 0, (int32_t)token);
        ds4_launch_pdl(tok_id_load_kernel, 1, 1, 0, 0,
                       g_tok_id_dev, (const int32_t *)(g_tok_slots_dev + 2 * slot), (const int32_t *)g_tok_next_dev);
        ds4_launch_pdl(embed_token_hc_dev_kernel, (n + 255) / 256, 256, 0, 0, (float *)out_hc->ptr, (const unsigned short *)wptr, g_tok_id_dev, n_vocab, n_embd, n_hc);
        return cuda_ok(cudaGetLastError(), "embed token dev launch");
    }
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

/* indexer 缓存恒 f16(core_gpu_graph.h)。只有 --quality 的 f32 直算核(单 token 直算 / 通用核)还要 f32 行: 转到一份
 * f32 暂存(按需增长, 非捕获态)。生产路(idx1/idxp)直接读 f16, 不经这里。 */
__global__ static void idx_f16_to_f32_kernel(float *out, const __half *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(x[i]);
}
static const float *indexer_rows_f32(const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t head_dim) {
    static float *tmp = NULL;
    static uint64_t cap = 0;
    const uint64_t need = (uint64_t)n_comp * head_dim;
    if (need > cap) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &cs);
        if (cs != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "ds4: indexer f32 view: 捕获态里要增长暂存(%u 行), --quality 下不该发生\n", n_comp);
            return NULL;
        }
        (void)cudaDeviceSynchronize();
        if (tmp) (void)cudaFree(tmp);
        tmp = NULL; cap = 0;
        if (cudaMalloc(&tmp, need * sizeof(float)) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
        cap = need;
    }
    idx_f16_to_f32_kernel<<<(unsigned)((need + 255u) / 256u), 256>>>(tmp, (const __half *)index_comp->ptr, need);
    return cuda_ok(cudaGetLastError(), "indexer f32 view convert launch") ? tmp : NULL;
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(uint16_t) ||   /* 缓存 f16 行 */
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u && !g_quality_mode) {
        /* 09-06: tensor-core 版(cuda_indexer_kernels_4), 1M 上下文 decode 的主账; --quality 走下面的 f32 直算。
         * 动态 shared 57.6 KB 超静态上限, 首发前抬一次属性(非流操作, 捕获态里也许可, 同 topk cub 路)。 */
        static int idx1_attr = 0;
        if (!idx1_attr) {
            if (cudaFuncSetAttribute(indexer_score_one_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)DS4_IDX1_SMEM) != cudaSuccess) {
                (void)cudaGetLastError();
                fprintf(stderr, "ds4: indexer wmma decode kernel: 动态 shared %zu B 抬不上去\n", DS4_IDX1_SMEM);
                return 0;
            }
            idx1_attr = 1;
        }
        indexer_score_one_wmma_kernel<<<(n_comp + DS4_IDX1_KEYS - 1u) / DS4_IDX1_KEYS, 256, DS4_IDX1_SMEM>>>(
            (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const __half *)index_comp->ptr, n_comp, pos0, ratio, scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one wmma launch");
    }
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u &&
        1) {
        const float *kf32 = indexer_rows_f32(index_comp, n_comp, head_dim);
        if (!kf32) return 0;
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         kf32,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (n_tokens >= 2u && n_tokens <= DS4_IDXT_QTOK && head_dim == 128u && n_head == 64u && !g_quality_mode) {
        /* 09-07 小批(投机 verify): token 放 N=8 的 mma.sync 核, 逐位复刻 idx1 的头内归约 ⇒ 与纯解码同集(见 kernels_4 尾注)。 */
        static int idxt_attr = 0;
        if (!idxt_attr) {
            if (cudaFuncSetAttribute(indexer_score_tokn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)DS4_IDXT_SMEM) != cudaSuccess) {
                (void)cudaGetLastError();
                fprintf(stderr, "ds4: indexer tokn mma kernel: 动态 shared %zu B 抬不上去\n", DS4_IDXT_SMEM);
                return 0;
            }
            idxt_attr = 1;
        }
        indexer_score_tokn_mma_kernel<<<(n_comp + DS4_IDXT_RANGE - 1u) / DS4_IDXT_RANGE, 256, DS4_IDXT_SMEM>>>(
            (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const __half *)index_comp->ptr, n_comp, n_tokens, pos0, ratio, scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score tokn mma launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u) {
        /* 09-06 idxp(任何批量): q 驻留 + K f16 流式(cuda_indexer_kernels_4), 直接读 f16 缓存, 捕获态也安全。
         * 旧的 wmma128/64/32 阶梯与 f32→f16 影子随缓存改 f16 一并删除。 */
        static int idxp_attr = 0;
        if (!idxp_attr) {
            int dev = 0, optin = 0;
            (void)cudaGetDevice(&dev);
            (void)cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            if (cudaFuncSetAttribute(indexer_scores_prefill_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)DS4_IDXP_SMEM) != cudaSuccess) {
                (void)cudaGetLastError();
                fprintf(stderr, "ds4: indexer prefill wmma kernel: 动态 shared %zu B 抬不上去(设备上限 %d B)\n",
                        DS4_IDXP_SMEM, optin);
                return 0;
            }
            idxp_attr = 1;
        }
        dim3 grid((n_comp + DS4_IDXP_RANGE - 1u) / DS4_IDXP_RANGE, (n_tokens + 15u) / 16u, 1);
        indexer_scores_prefill_wmma_kernel<<<grid, 256, DS4_IDXP_SMEM>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const __half *)index_comp->ptr, n_comp, n_tokens, pos0,
                                                     ratio, scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer scores prefill wmma launch");
    }
    const float *kf32 = indexer_rows_f32(index_comp, n_comp, head_dim);
    if (!kf32) return 0;
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         kf32,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 512u && n_comp <= 1024u &&
        1) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        1) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        1) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        1) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    /* 09-07: 解码/投机小批(≤8 token)先走多 block 版(kernels_6, 选集与顺序与下面逐字同); 暂存未建/大批走单 block 原路 */
    if (top_k == DS4_RTK_K && n_comp > 8192u && n_tokens <= DS4_RTKM_MAXTOK) {
        const int r = rtk_multi_launch((uint32_t *)selected->ptr, (const float *)scores->ptr, n_comp, n_tokens);
        if (r >= 0) return r;
    }
    /* 09-06: 大 n_comp 走 radix-select(cuda_indexer_kernels_5): 每 token 一 block 定阈值 + 定序压缩 + 512 元素排序,
     * 选集与顺序与下面的分块双调 + 树合并逐字同, 1M 时 decode 7.2 ms/token、prefill 2.6 s/块 的老账。 */
    if (top_k == DS4_RTK_K && n_comp > 8192u) {
        indexer_topk_radix512_kernel<<<n_tokens, DS4_RTK_THREADS>>>((uint32_t *)selected->ptr,
                                                                     (const float *)scores->ptr, n_comp, n_tokens);
        return cuda_ok(cudaGetLastError(), "indexer topk radix512 launch");
    }
    if (top_k == 512u &&
        1) {
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    DS4_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}
