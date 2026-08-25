/* core_model_map.c — model summary/find_tensor/加速器 span/tensor_data (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}

void model_summary(const ds4_model *m) {
    ds4_str name = {0};
    ds4_str arch = {0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t indexer_heads = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    model_get_u32(m, "deepseek4.block_count", &layers);
    model_get_u64(m, "deepseek4.context_length", &ctx_train);
    model_get_u32(m, "deepseek4.attention.head_count", &n_head);
    model_get_u32(m, "deepseek4.attention.head_count_kv", &n_head_kv);
    model_get_u32(m, "deepseek4.attention.key_length", &head_dim);
    model_get_u32(m, "deepseek4.attention.sliding_window", &n_swa);
    model_get_u32(m, "deepseek4.attention.indexer.head_count", &indexer_heads);
    model_get_u32(m, "deepseek4.attention.indexer.key_length", &indexer_head_dim);
    model_get_u32(m, "deepseek4.attention.indexer.top_k", &indexer_top_k);
    model_get_u32(m, "deepseek4.expert_count", &n_expert);
    model_get_u32(m, "deepseek4.expert_used_count", &n_expert_used);
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
        m->version, m->n_kv, m->n_tensors);
    if (layers) printf("layers: %u\n", layers);
    if (ctx_train) printf("train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        printf("attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (indexer_heads || indexer_head_dim || indexer_top_k) {
        printf("indexer: heads=%u head_dim=%u top_k=%u\n",
               indexer_heads, indexer_head_dim, indexer_top_k);
    }
    if (n_expert || n_expert_used || n_expert_groups || n_group_used) {
        printf("experts: count=%u used=%u groups=%u groups_used=%u\n",
               n_expert, n_expert_used, n_expert_groups, n_group_used);
    }
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);

    printf("tensor types:\n");
    for (uint32_t type = 0; type < sizeof(gguf_types)/sizeof(gguf_types[0]); type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            printf("  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(bytes);
            printf("\n");
        }
    }

}

ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

/* CUDA 默认 prefill 分块(见 ds4_default_prefill_cap_for_prompt 的实测注释); 0=未定。
 * 必须无条件声明: 使用点 ds4_default_prefill_cap_for_prompt 在所有构建里都编译,
 * 原先声明被圈进 #ifndef __APPLE__ 导致 Mac 构建 undeclared(stale .o 曾掩盖)。 */
int g_prefill_chunk_cuda = 0;

#ifndef DS4_NO_GPU
#ifndef __APPLE__
#ifdef DS4_CUDA_SPARK_HBM_CACHE
typedef struct {
    uint64_t off;
    uint64_t end;
} accelerator_tensor_span;

static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const accelerator_tensor_span *sa = a;
    const accelerator_tensor_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static uint64_t accelerator_cuda_preload_span_bytes(void) {
    uint64_t mb = 1024;
    const char *env = getenv("DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 64) mb = 64;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static bool accelerator_cache_model_tensor_spans(const ds4_model *m, uint64_t *cached_out) {
    /* Routed MoE expert weights (`*_exps.weight`) are ~65 GiB of the model on
     * V4-Flash but only top-K of N=256 experts fire per token — pre-caching
     * them in HBM wastes most of the budget on cold weights and starves the
     * hot non-MoE tensors that every token reads.  Skip them at the span-
     * build stage so the cap fills with attn / shared FFN / embedding /
     * output head.  Cold MoE expert reads fall back to the UVA-mapped
     * pointer. */
    accelerator_tensor_span *spans = xmalloc((size_t)m->n_tensors * sizeof(spans[0]));
    uint64_t nspan = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) {
            free(spans);
            return false;
        }
        /* Routed-expert weights are the only tensors with "_exps." in the
         * name; memmem is safe on short names (returns NULL).
         * ★DS4_CACHE_EXPERTS=1 收编专家★: 上面"缓存冷专家浪费预算"的前提是显存
         * 远小于模型(独显 24-80GB vs 89GB)。统一内存机器(GB10 121GiB)整模型装得下,
         * 此时跳过反而让每次专家读都跨 C2C 去 host 内存 —— 实测 GPU 利用率被按在 6%,
         * CPU 同时也闲着。开启需同时抬高 DS4_CUDA_WEIGHT_CACHE_LIMIT_GB(默认 24)。 */
        static int cache_exps = -1;
        if (cache_exps < 0) {
            const char *ce = getenv("DS4_CACHE_EXPERTS");
            if (ce && ce[0]) {
                cache_exps = (ce[0] == '1') ? 1 : 0;
            } else {
#ifdef DS4_CUDA_SPARK_HBM_CACHE
                /* GB10 默认收编专家(实测 decode 26.8→28.7): 内存账=模型+20GiB 余量
                 * 装得下才开, 装不下回退老策略(只缓 backbone) */
                const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
                const uint64_t total = (uint64_t)sysconf(_SC_PHYS_PAGES) * page;
                cache_exps = (m->size + 20ull * 1073741824ull <= total) ? 1 : 0;
#else
                cache_exps = 0;
#endif
            }
        }
        if (!cache_exps && memmem(t->name.ptr, t->name.len, "_exps.", 6) != NULL) {
            continue;
        }
        spans[nspan++] = (accelerator_tensor_span){
            .off = t->abs_offset,
            .end = t->abs_offset + t->bytes,
        };
    }
    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    uint64_t cached = 0;
    uint64_t merged = 0;
    for (uint64_t i = 0; i < nspan;) {
        /* group: contiguous tensors with gaps <= 64 KiB */
        const uint64_t g0 = i;
        uint64_t end = spans[i].end;
        i++;
        while (i < nspan && spans[i].off <= end + 65536u) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        /* Pack the group's tensors into chunks WITHOUT splitting any single
         * tensor: a device-side consumer (e.g. the VQ expert blob relocation)
         * needs each tensor's bytes virtually contiguous in one cudaMalloc
         * block.  A tensor larger than max_span becomes its own whole chunk
         * (the arena allocator grows to fit). */
        uint64_t j = g0;
        while (j < i) {
            const uint64_t off = spans[j].off;
            uint64_t chunk_end = spans[j].end;
            j++;
            while (j < i && spans[j].end - off <= max_span) {
                if (spans[j].end > chunk_end) chunk_end = spans[j].end;
                j++;
            }
            char label[96];
            snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
            if (ds4_gpu_cache_model_range(m->map, m->size, off, chunk_end - off, label) == 0) {
                fprintf(stderr,
                        "ds4: accelerator failed to cache model tensor span %" PRIu64
                        " at offset %" PRIu64 "\n",
                        merged, off);
                free(spans);
                return false;
            }
            cached += chunk_end - off;
            merged++;
        }
    }
    free(spans);
    if (cached_out) *cached_out = cached;
    return true;
}
#endif

bool accelerator_cache_model_tensors(ds4_backend backend, const ds4_model *m) {
    if (backend == DS4_BACKEND_CUDA && g_prefill_chunk_cuda == 0) g_prefill_chunk_cuda = 256;
    if (backend != DS4_BACKEND_CUDA) return true;
    if (!m || !m->map || m->size == 0) return false;
    if (getenv("DS4_CUDA_DIRECT_MODEL") != NULL) {
        return true;
    }

#ifdef DS4_CUDA_SPARK_HBM_CACHE
    const double t0 = now_sec();
    uint64_t cached = 0;
    if (!accelerator_cache_model_tensor_spans(m, &cached)) return false;
#else
    uint64_t cached = 0;
#endif
    if (getenv("DS4_CUDA_Q8_F16_PRELOAD") != NULL ||
        getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL) {
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const ds4_tensor *t = &m->tensors[i];
            if (t->bytes == 0) continue;
            if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) return false;
            char label[128];
            snprintf(label, sizeof(label), "tensor:%.*s", (int)t->name.len, t->name.ptr);
            if (t->type == DS4_TENSOR_Q8_0 && t->ndim == 2 &&
                ds4_gpu_cache_q8_f16_range(m->map, m->size, t->abs_offset, t->bytes, t->dim[0], t->dim[1], label) == 0) {
                fprintf(stderr, "ds4: accelerator failed to cache dequantized Q8 tensor %.*s\n",
                        (int)t->name.len, t->name.ptr);
                return false;
            }
        }
    }
#ifdef DS4_CUDA_SPARK_HBM_CACHE
    {   /* q8 repack 预建(decode gemv 快路): 必须先于 token graph capture */
        uint64_t q8r_bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const ds4_tensor *t = &m->tensors[i];
            if (t->type != DS4_TENSOR_Q8_0 || t->ndim != 2 || t->bytes == 0) continue;
            if (memmem(t->name.ptr, t->name.len, "_exps.", 6) != NULL) continue;
            if (ds4_gpu_q8r_preload(m->map, m->size, t->abs_offset, t->dim[0], t->dim[1]))
                q8r_bytes += t->bytes;
        }
        if (q8r_bytes)
            fprintf(stderr, "ds4: CUDA q8 repack preloaded %.2f GiB\n",
                    (double)q8r_bytes / 1073741824.0);
    }
    if (cached != 0) {
        const double t1 = now_sec();
        if (ds4_log_is_tty(stderr)) fputc('\n', stderr);
        fprintf(stderr,
                "ds4: CUDA startup model cache prepared %.2f GiB of tensor spans in %.3fs\n",
                (double)cached / 1073741824.0,
                t1 - t0);
    }
#else
    (void)cached;
#endif
    return true;
}
#else
bool accelerator_cache_model_tensors(ds4_backend backend, const ds4_model *m) {
    (void)backend;
    (void)m;
    return true;
}
#endif
#endif

/* Return the in-place tensor payload inside the mapped GGUF. */
const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return m->map + t->abs_offset;
}

