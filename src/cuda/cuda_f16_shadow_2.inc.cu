/* cuda_f16_shadow_2.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * host 探针辅助 + 全q2 f16 影子(f16 专线家族权重装载时一次性 dequant→f16 device buffer)。
 */
static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (0) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    if (0) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return 1;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return 1;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (1 &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_use_dp4a(void) {
    return 1;
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        1 &&
        1) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (0) return 0;
    if (0) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return 0;
    }
    return 0 &&
           in_dim == 1024u && out_dim == 32768u;
}

static std::unordered_set<uint64_t> g_q8_f16_denied; /* sticky: weight stays on q8 kernels all run */

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (g_q8_f16_denied.count(offset)) return NULL;
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) {
        g_q8_f16_denied.insert(offset);
        return NULL;
    }

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        g_q8_f16_denied.insert(offset);
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    /* The dequant runs on the default stream while consumers may issue work on
     * other streams (or capture a graph) immediately after this returns.  The
     * cache entry is permanent, so publishing it before the fill completes
     * poisons every later token.  One synchronize per weight, at build time
     * only. */
    if (!cuda_ok(cudaDeviceSynchronize(), "q8 fp16 dequant sync")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant sync failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    return dev;
}

/* 只读查表版(decode/graph-capture 安全): 不建 cache, 命中已有条目才返回 */
static const __half *cuda_q8_f16_lookup(const void *model_map, uint64_t offset,
                                        uint64_t weight_bytes, uint64_t in_dim, uint64_t out_dim) {
    auto it = g_q8_f16_by_offset.find(offset);
    if (it == g_q8_f16_by_offset.end()) return NULL;
    const cuda_q8_f16_range &r = g_q8_f16_ranges[it->second];
    if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
        r.in_dim == in_dim && r.out_dim == out_dim) return r.device_ptr;
    return NULL;
}

