/* metal_model_map.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out_hc) < out_bytes) {
            fprintf(stderr, "ds4: Metal graph embedding received undersized HC output buffer\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger row_bytes = (NSUInteger)n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             row_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token;
        const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
        const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
        ds4_gpu_get_rows_args args = {
            .ne00t = (int32_t)n_embd,
            .ne00 = (int32_t)n_embd,
            .nb01 = src_row_bytes,
            .nb02 = (uint64_t)n_vocab * src_row_bytes,
            .nb03 = (uint64_t)n_vocab * src_row_bytes,
            .ne10 = 1,
            .nb10 = sizeof(int32_t),
            .nb11 = sizeof(int32_t),
            .nb12 = sizeof(int32_t),
            .nb1 = dst_row_bytes,
            .nb2 = dst_row_bytes,
            .nb3 = dst_row_bytes,
        };
        NSUInteger nth = (NSUInteger)n_embd;
        const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > max_threads) nth = max_threads;
        if (nth == 0) nth = 1;
        const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_get_rows_f16_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBytes:&token_i32 length:sizeof(token_i32) atIndex:2];
        [enc setBuffer:g_embed_rows_buffer offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nw0, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * n_hc * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out_hc) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal graph batched embedding received undersized buffers\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger rows_bytes = (NSUInteger)n_tokens * n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             rows_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_f16(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           g_embed_rows_buffer,
                                           0,
                                           n_vocab,
                                           n_tokens,
                                           n_embd) ||
            !ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  n_tokens,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed tokens")) return 0;
    }

    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;
    max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(map_size, max_tensor_bytes);

    @autoreleasepool {
        if (g_model_map_ptr == model_map &&
            g_model_map_size == model_size &&
            g_model_mapped_offset == map_offset &&
            g_model_mapped_size == map_size &&
            g_model_mapped_max_tensor_bytes == max_tensor_bytes) {
            return 1;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            if (g_model_views[i].model_map == model_map &&
                g_model_views[i].model_size == model_size &&
                map_offset >= g_model_views[i].model_offset &&
                map_offset + map_size <= g_model_views[i].model_offset + g_model_views[i].bytes) {
                return 1;
            }
        }

        ds4_gpu_model_residency_clear();
        if (!ds4_gpu_map_model_views(model_map, model_size, map_offset, map_size, max_tensor_bytes)) {
            ds4_gpu_model_residency_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        if (model_size > g_efetch_model_size) g_efetch_model_size = model_size;
        g_model_mapped_offset = map_offset;
        g_model_mapped_size = map_size;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        fprintf(stderr,
                "ds4: Metal mapped mmaped model as %u overlapping shared buffers\n",
                g_model_view_count);
        return 1;
    }
}

/* Shared implementation for the plain and split span loaders. resident_flags is
 * an optional per-span bool array: when NULL every span is resident (legacy
 * behaviour); otherwise a false entry wraps that span's views without adding them
 * to the model residency set, so their clean file-backed pages stay reclaimable
 * (routed-expert offload). */
static int ds4_gpu_set_model_map_spans_impl(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        const bool *resident_flags,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    /* A single all-resident span is exactly the contiguous range loader; skip the
     * disjoint-buffer bookkeeping. A non-resident single span still needs the
     * per-view resident_hint, so fall through to the general path for it. */
    if (count == 1 && (!resident_flags || resident_flags[0])) {
        return ds4_gpu_set_model_map_range(model_map,
                                           model_size,
                                           offsets[0],
                                           sizes[0],
                                           max_tensor_bytes);
    }

    @autoreleasepool {
        const double t0 = ds4_gpu_now_ms();
        max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(model_size, max_tensor_bytes);

        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();

        uint64_t mapped_total = 0;
        uint64_t first_offset = UINT64_MAX;
        for (uint32_t i = 0; i < count; i++) {
            if (offsets[i] > model_size || sizes[i] == 0 || sizes[i] > model_size - offsets[i]) {
                fprintf(stderr, "ds4: Metal model span %u is outside the GGUF mapping\n", i);
                ds4_gpu_model_residency_clear();
                ds4_gpu_model_views_clear();
                return 0;
            }
            if (offsets[i] < first_offset) first_offset = offsets[i];
            uint64_t effective_max = max_tensor_bytes;
            if (effective_max > sizes[i]) effective_max = sizes[i];
            const bool span_resident = resident_flags ? resident_flags[i] : true;
            if (!ds4_gpu_add_model_view_range(model_map,
                                              model_size,
                                              offsets[i],
                                              sizes[i],
                                              effective_max,
                                              span_resident,
                                              &mapped_total)) {
                ds4_gpu_model_residency_clear();
                ds4_gpu_model_views_clear();
                return 0;
            }
        }
        if (!ds4_gpu_finish_model_views(t0, mapped_total, first_offset)) {
            ds4_gpu_model_residency_clear();
            ds4_gpu_model_views_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        if (model_size > g_efetch_model_size) g_efetch_model_size = model_size;
        g_model_mapped_offset = first_offset == UINT64_MAX ? 0 : first_offset;
        g_model_mapped_size = mapped_total;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        fprintf(stderr,
                "ds4: Metal mapped mmaped model as %u disjoint shared buffers across %u tensor spans\n",
                g_model_view_count,
                count);
        return 1;
    }
}

int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    return ds4_gpu_set_model_map_spans_impl(model_map, model_size, offsets, sizes,
                                            NULL, count, max_tensor_bytes);
}

/* Reduced-memory loader: resident spans (backbone) are wrapped and added to the
 * residency set; non-resident spans (routed experts) are wrapped so the hot path
 * still resolves a buffer, but are kept out of the residency set so their clean
 * file-backed pages stay reclaimable. Concatenate the two span lists, mark the
 * resident ones true and the rest false, and pass one array. */
int ds4_gpu_set_model_map_spans_split(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        const bool *resident_flags,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    if (!resident_flags) return 0;   /* split loader requires explicit flags */
    return ds4_gpu_set_model_map_spans_impl(model_map, model_size, offsets, sizes,
                                            resident_flags, count, max_tensor_bytes);
}

/* GPU 跨度计时: Metal 侧不做事件对(CUDA 专用归因工具), 返回 -1 表示不可用。 */
int ds4_gpu_register_aux_model_map(const void *map, uint64_t size) { (void)map; (void)size; return 1; }
/* q8 f16 影子缓存是 CUDA 的 cuBLAS 小批路专用(Metal 稠密核直接吃 q8_0), 预建入口空实现(drafter 绑定时调, 09-07) */
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes,
                               uint64_t in_dim, uint64_t out_dim, const char *label) {
    (void)model_map; (void)model_size; (void)offset; (void)bytes; (void)in_dim; (void)out_dim; (void)label; return 1;
}

int ds4_gpu_sanitize_finite_tensor(ds4_gpu_tensor *t, uint64_t n) { (void)t; (void)n; return 1; }

int ds4_gpu_sanitize_router_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *w, uint32_t n, uint32_t ne) {
    (void)sel; (void)w; (void)n; (void)ne; return 1;   /* Metal: 路由 kernel 不产 -1 */
}

int ds4_gpu_batch_graph_begin(int slot) { (void)slot; return 0; }

/* Metal: 无批图 */
int ds4_gpu_batch_graph_end_launch(int encode_ok) { (void)encode_ok; return 0; }

int ds4_gpu_matmul_q2_K_pair_batch_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                          const void *model_map, uint64_t model_size,
                                          uint64_t off0, uint64_t off1,
                                          uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
                                          const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out0; (void)out1; (void)model_map; (void)model_size; (void)off0; (void)off1;
    (void)in_dim; (void)out0_dim; (void)out1_dim; (void)x; (void)n_tok;
    return 0;   /* Metal: 调用方回退两次单发 */
}

void ds4_gpu_span_begin(void) {}

float ds4_gpu_span_end(void) { return -1.0f; }

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map_range(model_map, model_size, 0, model_size, 0);
}

/* Main-model GGUF file descriptor, registered by the engine right after the
 * model is opened (the whole file is mmapped from offset 0, so file offset ==
 * map offset).  The P1.1 single-copy expert gather (project.md) preads cold
 * expert bytes straight from this fd into the Shared
 * scratch MTLBuffer, bypassing the mmap page-fault + memcpy double copy. */
int g_model_fd = -1;

int g_model_fd_conflict = 0;

int ds4_gpu_set_model_fd(int fd) {
    if (fd < 0) return 1;
    if (g_model_fd >= 0 && g_model_fd != fd) {
        /* Two different model files registered in one process: the pread fast
         * path can no longer tell which file backs a given map; disable it. */
        g_model_fd_conflict = 1;
    }
    g_model_fd = fd;
    return 1;
}

id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) {
            *inner_offset = offset - view_start;
            return g_model_views[i].buffer;
        }
    }

    fprintf(stderr,
            "ds4: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            ds4_gpu_gib(offset),
            ds4_gpu_gib(end));
    return nil;
}
