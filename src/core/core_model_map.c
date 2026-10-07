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

#ifndef DS4_NO_GPU
#ifndef __APPLE__
typedef struct {
    uint64_t off;
    uint64_t end;
    /* 0 = 主干骨架(每步必读, 永远先拷)。后面两档谁先谁后看投机开不开:
     * 投机关: 1 主干专家/blob, 3 DSpark 三塔(一次都不读, 排最后)
     * 投机开: 1 DSpark 三塔(每轮都读, 只有 7.3 GiB), 2 主干专家/blob(98 GiB, 挤掉尾巴只丢 3%) */
    uint32_t prio;
    /* 1 = VQ 专家 blob: 自成一段(不与邻居合并), 走 ds4_gpu_cache_vq_blob —— 它装载时把载荷挪到位流 128 B 对齐的位置,
     * 重写的是 blob 自己的槽表, 段里要是混进别的张量, 它们的设备地址就跟着错位(见 cuda_vq_align.inc.cu)。 */
    uint32_t blob;
} accelerator_tensor_span;

static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const accelerator_tensor_span *x = a, *y = b;
    if (x->prio != y->prio) return x->prio < y->prio ? -1 : 1;   /* 骨架优先 */
    if (x->off != y->off) return x->off < y->off ? -1 : 1;
    return 0;
}

/* 装载期的内存地板(single.md S1 §4 第一条): 拷进设备副本的那一刻, 源 mmap 页还没被回收,
 * 同一份字节短暂占两份 —— 实测 109.79 GiB 装完时 MemAvailable 谷底 10.5 GB, 只比地板(10 GB, 09-08
 * 实撞 5 GB 时 swap、解码 21.6 → 12.5)高 0.5 GB。所以每装一段就看一眼真实余量, 低于 12 GB 就收手:
 * 剩下的张量走主机映射(慢, 但对账会大声报), 总比把机器推进 swap 强。★别 OOM 是最高约束★ */
#define DS4_CACHE_AVAIL_FLOOR_BYTES (12ull * 1000000000ull)   /* 十进制 GB, 与 /proc/meminfo 同口径 */
static uint64_t accelerator_mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return UINT64_MAX;   /* 读不到就不闸(别把功能建在探测上) */
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %llu kB", (unsigned long long *)&kb) == 1) break;
    fclose(f);
    return kb ? kb * 1024ull : UINT64_MAX;
}

static uint64_t accelerator_cuda_preload_span_bytes(void) {
    /* 1 GiB/段: 单段 cudaMalloc 足够大到吃满预载带宽, 又不至于让 arena
     * 一次性要走巨块(超大张量自成整段, 见下方分组逻辑)。 */
    return 1024ull * 1048576ull;
}

static bool accelerator_cache_model_tensor_spans(const ds4_model *m, uint64_t *cached_out, uint64_t *want_out) {
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
         * ★收编专家★: 上面"缓存冷专家浪费预算"的前提是显存远小于模型
         * (独显 24-80GB vs 89GB)。统一内存机器(GB10 121GiB)整模型装得下,
         * 此时跳过反而让每次专家读都跨 C2C 去 host 内存 —— 实测 GPU 利用率
         * 被按在 6%, CPU 同时也闲着。
         * 2026-10-07 前这是编译宏 DS4_CUDA_SPARK_HBM_CACHE(make cuda-spark 才开): 同一份源码用
         * cuda-generic 编出来放到 GB10 上就退化成"只缓骨架"。现在按设备属性运行时判
         * (ds4_gpu_unified_memory_host: GPU 经主机页表访问整机内存), 独显仍只缓骨架。 */
        static int cache_exps = -1;
        if (cache_exps < 0) {
            /* 统一内存机器默认收编专家(实测 decode 26.8→28.7): 内存账=模型+20GiB 余量
             * 装得下才开, 装不下回退老策略(只缓 backbone) */
            const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
            const uint64_t total = (uint64_t)sysconf(_SC_PHYS_PAGES) * page;
            cache_exps = (ds4_gpu_unified_memory_host() && m->size + 20ull * 1073741824ull <= total) ? 1 : 0;
        }
        const bool is_exp = memmem(t->name.ptr, t->name.len, "_exps.", 6) != NULL;
        const bool is_blob = memmem(t->name.ptr, t->name.len, "_exps_vq.", 9) != NULL;   /* 合一 VQ blob(V4/V4.1 专家字节) */
        if (!cache_exps && is_exp) {
            continue;
        }
        /* ★2026-09-15 single.md S1: 三塔排最后★ —— `mtp.*` 是 DSpark 草稿器的 7.3 GiB,
         * 它的名字里既没有 "_exps." 也没有 "_exps_vq.", 原来算 prio 0 跟骨架抢在最前面装。
         * 而投机没开(--no-dspark, 本底座默认)时这 7.3 GiB 一次都不会被读, 却把主干最后几层的
         * 专家 blob 挤出了预算 —— 挤出去的那几层每步要多付 5~25 ms(见 single.md §2.1)。
         *
         * ★2026-09-16 mtp-1.md M3′: 投机开着时反过来排到专家前面★
         * 模型 109.79 GiB 装不完(装进 106.51, 余 3.28 GiB 走主机映射, 那 3.28 GiB 会被 kswapd
         * 一直回收、每次读都要缺页), 所以问题不是"能不能全装下", 是**让谁去当那 3.28 GiB**。
         * 一轮投机的读量: 三塔 1.41 GB, 主干专家 1.61 GB ×(1+k)。三塔总共才 7.3 GiB,
         * 3.28 GiB 落在它身上 = 45% 的草稿读要缺页; 落在 98 GiB 的主干专家上只有 3.3%。
         * ⇒ 投机开着就把三塔提到与专家同级(按偏移顺序混排, 谁在后面谁被挤), 盘上实测的签名是
         *   塔核"中位 1.1 ms / 最大 21.7 ms"那条长尾。★别把它提到 0★: 骨架是每步必读的, 不能让。 */
        const bool is_mtp = t->name.len > 4 && memcmp(t->name.ptr, "mtp.", 4) == 0;
        /* ★同级不够, 必须把三塔排到专家**前面**★(2026-09-16 实撞): 先试过"投机开着就把 mtp 从 2 降到 1",
         * 读数一点没动 —— 因为同一优先级内是按**文件偏移**排的, 而 mtp.* 是转换器追加在文件最末尾的,
         * 排在所有专家 blob 后面, 照样是被挤出去的那 3.28 GiB。要它进设备就得自己占一档。 */
        const uint32_t mtp_prio = g_ds4_v41_dspark ? 1u : 3u;
        const uint32_t exp_prio = g_ds4_v41_dspark ? 2u : 1u;
        spans[nspan++] = (accelerator_tensor_span){
            .off = t->abs_offset,
            .end = t->abs_offset + t->bytes,
            .prio = is_mtp ? mtp_prio : ((is_exp || is_blob) ? exp_prio : 0u),
            .blob = is_blob ? 1u : 0u,
        };
    }
    /* ★骨架先拷、专家/blob 后填(2026-09-12)★: 之前纯按偏移排, V4.1 的 103 GiB 里 98 GB 是 40 层 blob, 预算(当时是总内存-24 GiB, 现为 -8 GiB)
     * 被前面的 blob 吃光, 排在文件最末的 output.weight(350 MB)没拷进设备, 只能经 cudaHostRegister 的文件映射读 ——
     * 实测该映射尾页首次被 GPU 读偶发返回垃圾(e8m0 垃圾 → 2^128 → f16 inf → 整列 logits NaN, 重读即对)。
     * 骨架(每 token 都读、含 head)必须常驻; blob 再按偏移顺序填到预算为止, 余下走映射。 */
    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    uint64_t total_bytes = 0;
    for (uint64_t i = 0; i < nspan; i++) total_bytes += spans[i].end - spans[i].off;
    uint64_t cached = 0;
    uint64_t merged = 0;
    for (uint64_t i = 0; i < nspan;) {
        /* group: contiguous tensors with gaps <= 64 KiB */
        const uint64_t g0 = i;
        uint64_t end = spans[i].end;
        i++;
        while (i < nspan && !spans[g0].blob && !spans[i].blob && spans[i].off <= end + 65536u) {
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
            if (accelerator_mem_available_bytes() < DS4_CACHE_AVAIL_FLOOR_BYTES) {
                fprintf(stderr,
                        "ds4: ★内存地板★ MemAvailable 已低于 %.0f GB, 停止把权重拷进设备副本;\n"
                        "     余下 %.2f GiB 走主机映射(每步会有长尾, 见 single.md §2.1)\n",
                        (double)DS4_CACHE_AVAIL_FLOOR_BYTES / 1e9,
                        (double)(total_bytes - cached) / 1073741824.0);
                free(spans);
                if (cached_out) *cached_out = cached;
                if (want_out) *want_out = total_bytes;
                return true;
            }
            const int ok = spans[g0].blob ? ds4_gpu_cache_vq_blob(m->map, m->size, off, chunk_end - off, label)
                                          : ds4_gpu_cache_model_range(m->map, m->size, off, chunk_end - off, label);
            if (ok == 0) {
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
    if (want_out) *want_out = total_bytes;
    return true;
}

bool accelerator_cache_model_tensors(ds4_backend backend, const ds4_model *m) {
    if (backend != DS4_BACKEND_CUDA) return true;
    if (!m || !m->map || m->size == 0) return false;

    const double t0 = now_sec();
    uint64_t cached = 0, want = 0;
    if (!accelerator_cache_model_tensor_spans(m, &cached, &want)) return false;
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
        /* ★对账(single.md S1)★: `cached` 是我们**请求**装的字节, `ds4_gpu_model_cache_bytes()` 是
         * 真正拷进设备副本的。超预算的段是静默走主机映射的 —— 那几 GiB 就是每步 5~25 ms 的长尾,
         * 而以前的日志一个字都看不见。差值 >0 必须大声说, 否则后面每一刀的速度读数都被它糊掉。 */
        const uint64_t in_dev = ds4_gpu_model_cache_bytes();
        /* ★分母是"该装的全部"(want), 不是"这趟装成了的"(cached)★ —— 拿 cached 当分母, 闸一提前收手
         * 差值就恒为 0, 对账等于没做(2026-09-15 第一版就这么写错过)。 */
        const uint64_t mapped = want > in_dev ? want - in_dev : 0;
        fprintf(stderr,
                "ds4: CUDA startup model cache prepared %.2f GiB of tensor spans in %.3fs\n",
                (double)cached / 1073741824.0,
                t1 - t0);
        fprintf(stderr, "ds4: [缓存对账] 装进设备 %.2f GiB / 走主机映射 %.2f GiB\n",
                (double)in_dev / 1073741824.0, (double)mapped / 1073741824.0);
        if (mapped > 0)
            fprintf(stderr,
                    "ds4: ★%.2f GiB 权重没进设备副本, 走 cudaHostRegister 映射 —— 若解码路会读它, 每步要被页回收\n"
                    "     拖出几毫秒到几十毫秒的长尾(single.md §2.1)。被挤出的是最后装的那批(装载优先级:\n"
                    "     主干骨架 > 主干专家 blob > DSpark 三塔), 所以三塔被挤出时解码不受影响(它一次都不读)。★\n",
                    (double)mapped / 1073741824.0);
    }
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


/* 09-07(1M 投机剖面): drafter 的 q8_0 投影走 cuBLAS f16 影子 GEMM, 影子原本在首个投机轮里才懒建(dequant + cudaMalloc ~0.5 s),
 * 短尺被首轮拖 10%。这里按后端运行时同一形状规则预建(不满足规则的张量后端自己跳过, Metal 是空实现), 把一次性开销挪到加载期。
 * 主模型的影子随 prefill 首块建, 不在这里。 */
void model_preload_q8_f16_shadows(const ds4_model *m) {
#ifndef DS4_NO_GPU
    if (!m || !m->tensors) return;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->type != DS4_TENSOR_Q8_0 || t->ndim != 2 || t->bytes == 0) continue;
        if (memmem(t->name.ptr, t->name.len, "_exps.", 6) != NULL) continue;
        (void)ds4_gpu_cache_q8_f16_range(m->map, m->size, t->abs_offset, t->bytes, t->dim[0], t->dim[1], "q8_0");
    }
#else
    (void)m;
#endif
}
