/* metal_zchain.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* ===== go-onebit DQZ2 zchain (ds4_gpu.h contract) =====
 * Small resident tables uploaded once at load; per-layer dispatch metadata
 * (op ranges, GE presence) stays host-side in these statics. */
static id<MTLBuffer> g_zchain_ops;

/* [n_ops_total][16] f32 */
static id<MTLBuffer> g_zchain_v8;

/* [n_blk][8][d_model] fp16 (1-elem dummy when none) */
static id<MTLBuffer> g_zchain_ge;

/* [n_layer][n_expert] f32 (1-elem dummy when none) */
static uint32_t     *g_zchain_layer_off;

/* [n_layer+1] */
static uint8_t      *g_zchain_ge_present;

/* [n_layer] */
static uint32_t      g_zchain_n_layer, g_zchain_n_expert, g_zchain_d_model;

/* frozen z^L (type 6, 2026-07-14): packed fp16 factors + per-layer host meta */
static id<MTLBuffer> g_zchain_zlm;

/* concat z[k]|U[d*k]|V[d*k] per zl layer (1-elem dummy when none) */
static uint32_t     *g_zchain_zl_off;

/* [n_layer] offset in halves */
static uint32_t     *g_zchain_zl_k;

/* [n_layer] rank (0 = absent) */
static float        *g_zchain_zl_tr;

/* [n_layer] trust-region factor */
static uint32_t     *g_zchain_zl_din;

/* [n_layer] V input dim (0/d=linear, 3d=ftA md86) */

/* dense/attn_output Q2_K(2026-08-19 全q2 基座): Metal kernel 未实现 — 明确拒绝,
 * 禁按 q8 语义静默跑错。全q2 模型当前只在 CUDA(spark) 服役。 */
int ds4_gpu_register_q2k_f16_shadow(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t rows, uint64_t cols) {
    (void)model_map; (void)model_size; (void)offset; (void)rows; (void)cols;
    fprintf(stderr, "ds4: 全q2 f16 影子 Metal 未实现(全q2 模型请用 CUDA)\n");
    return 0;
}

int ds4_gpu_matmul_q2_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    fprintf(stderr, "ds4: dense Q2_K matmul Metal 未实现(全q2 模型请用 CUDA)\n");
    return 0;
}

int ds4_gpu_attention_output_q2k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out; (void)low; (void)model_map; (void)model_size; (void)out_a_offset;
    (void)out_b_offset; (void)group_dim; (void)rank; (void)n_groups; (void)out_dim;
    (void)heads; (void)n_tokens;
    fprintf(stderr, "ds4: attn_output Q2_K Metal 未实现(全q2 模型请用 CUDA)\n");
    return 0;
}

int ds4_gpu_zchain_rte_set(const uint16_t *rm, const uint32_t *off, const uint32_t *k,
                           const float *scale, uint32_t n_layer, uint32_t n_expert,
                           uint64_t total_halves) {
    (void)rm; (void)off; (void)scale; (void)n_expert; (void)total_halves;
    /* 路由闭式侧车(type8) Metal kernel 未实现: 有侧车层 → 缴械并明示(禁静默跑错)。 */
    if (k) for (uint32_t l = 0; l < n_layer; l++) if (k[l]) {
        fprintf(stderr, "ds4: zchain route 侧车(type8) Metal 未实现 — 路由侧车缴械(路由=裸量化)\n");
        return 1;
    }
    return 1;
}

int ds4_gpu_zchain_route_bias(ds4_gpu_tensor *logits, const ds4_gpu_tensor *x,
                              uint32_t layer, uint32_t n_tokens) {
    (void)logits; (void)x; (void)layer; (void)n_tokens;
    return 1;   /* 未上传即无侧车, 零成本通过 */
}

int ds4_gpu_zchain_zl_set(const uint16_t *zlm, const uint32_t *off, const uint32_t *k,
                          const uint32_t *din, const float *tr, const uint32_t *mul,
                          uint32_t n_layer, uint64_t total_halves) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    /* 乘性侧车(type7 AMP / type9 AMPD 动态 z) 的 Metal kernel 未实现:
     * kernel_dsv4_zchain_scale 只有加性 z^L 那条路(参数结构里连 zl_mul 都没有)。
     * 有 mul 层 → 整族缴械并明示(宁可不放大, 禁按加性语义静默跑错)。 */
    if (mul) for (uint32_t l = 0; l < n_layer; l++) if (mul[l]) {
        fprintf(stderr, "ds4: zchain 乘性侧车(type7 AMP / type9 AMPD) Metal 未实现 "
                        "— z^L 侧车整族缴械(输出=裸量化)\n");
        return 1;
    }
    @autoreleasepool {
        free(g_zchain_zl_off); g_zchain_zl_off = NULL;
        free(g_zchain_zl_k);   g_zchain_zl_k = NULL;
        free(g_zchain_zl_tr);  g_zchain_zl_tr = NULL;
        free(g_zchain_zl_din); g_zchain_zl_din = NULL;
        g_zchain_zlm = nil;
        if (!n_layer || !k || !off || !tr) return 1;   /* nothing to upload = ok */
        g_zchain_zlm = (zlm && total_halves)
            ? [g_device newBufferWithBytes:zlm
                                    length:(NSUInteger)(total_halves * sizeof(uint16_t))
                                   options:MTLResourceStorageModeShared]
            : [g_device newBufferWithLength:sizeof(uint16_t)
                                    options:MTLResourceStorageModeShared];
        if (!g_zchain_zlm) return 0;
        g_zchain_zl_off = malloc((size_t)n_layer * sizeof(uint32_t));
        g_zchain_zl_k   = malloc((size_t)n_layer * sizeof(uint32_t));
        g_zchain_zl_tr  = malloc((size_t)n_layer * sizeof(float));
        if (!g_zchain_zl_off || !g_zchain_zl_k || !g_zchain_zl_tr) return 0;
        memcpy(g_zchain_zl_off, off, (size_t)n_layer * sizeof(uint32_t));
        memcpy(g_zchain_zl_k,   k,   (size_t)n_layer * sizeof(uint32_t));
        memcpy(g_zchain_zl_tr,  tr,  (size_t)n_layer * sizeof(float));
        if (din) {
            g_zchain_zl_din = malloc((size_t)n_layer * sizeof(uint32_t));
            if (!g_zchain_zl_din) return 0;
            memcpy(g_zchain_zl_din, din, (size_t)n_layer * sizeof(uint32_t));
        }
        uint32_t nz = 0;
        for (uint32_t l = 0; l < n_layer; l++) if (k[l]) nz++;
        if (nz) fprintf(stderr, "ds4: Metal zchain z^L resident: %u layers\n", nz);
    }
    return 1;
}

int ds4_gpu_zchain_set(
        const float *ops, const uint32_t *layer_off, const uint16_t *v8,
        const float *ge, const uint8_t *ge_present,
        uint32_t n_layer, uint32_t n_expert, uint32_t d_model,
        uint32_t n_ops_total, uint32_t n_v8_blocks) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!layer_off || n_layer == 0 || n_expert == 0 || d_model == 0) return 0;
    @autoreleasepool {
        g_zchain_ops = n_ops_total
            ? [g_device newBufferWithBytes:ops
                                    length:(NSUInteger)n_ops_total * 16u * sizeof(float)
                                   options:MTLResourceStorageModeShared]
            : [g_device newBufferWithLength:16u * sizeof(float)
                                    options:MTLResourceStorageModeShared];
        g_zchain_v8 = (v8 && n_v8_blocks)
            ? [g_device newBufferWithBytes:v8
                                    length:(NSUInteger)n_v8_blocks * 8u * d_model * sizeof(uint16_t)
                                   options:MTLResourceStorageModeShared]
            : [g_device newBufferWithLength:sizeof(uint16_t)
                                    options:MTLResourceStorageModeShared];
        g_zchain_ge = (ge && ge_present)
            ? [g_device newBufferWithBytes:ge
                                    length:(NSUInteger)n_layer * n_expert * sizeof(float)
                                   options:MTLResourceStorageModeShared]
            : [g_device newBufferWithLength:sizeof(float)
                                    options:MTLResourceStorageModeShared];
        if (!g_zchain_ops || !g_zchain_v8 || !g_zchain_ge) return 0;
        free(g_zchain_layer_off); free(g_zchain_ge_present);
        g_zchain_layer_off  = malloc((size_t)(n_layer + 1) * sizeof(uint32_t));
        g_zchain_ge_present = calloc(n_layer, 1);
        if (!g_zchain_layer_off || !g_zchain_ge_present) return 0;
        memcpy(g_zchain_layer_off, layer_off, (size_t)(n_layer + 1) * sizeof(uint32_t));
        if (ge_present) memcpy(g_zchain_ge_present, ge_present, n_layer);
        g_zchain_n_layer = n_layer; g_zchain_n_expert = n_expert; g_zchain_d_model = d_model;
        fprintf(stderr, "ds4: Metal zchain resident: %u ops, %u dyn8 blocks, GE %s\n",
                n_ops_total, n_v8_blocks, (ge && ge_present) ? "yes" : "no");
    }
    return 1;
}

int ds4_gpu_zchain_ge_apply(
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint32_t layer, uint32_t n_expert_used, uint32_t n_tokens) {
    if (!g_zchain_n_layer || layer >= g_zchain_n_layer) return 1;   /* not loaded => no-op */
    if (!g_zchain_ge_present[layer]) return 1;
    if (!weights || !selected || n_expert_used == 0 || n_tokens == 0) return 0;
    @autoreleasepool {
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> sbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t total = (uint64_t)n_tokens * n_expert_used;
        if (!wbuf || !sbuf ||
            ds4_gpu_tensor_bytes(weights) < total * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < total * sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal zchain ge received undersized buffers\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_dsv4_zchain_ge");
        if (!pipeline) return 0;
        struct { uint32_t n_expert, n_expert_used, n_tokens, ge_base; } args = {
            g_zchain_n_expert, n_expert_used, n_tokens, layer * g_zchain_n_expert
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:1];
        [enc setBuffer:sbuf offset:ds4_gpu_tensor_offset(selected) atIndex:2];
        [enc setBuffer:g_zchain_ge offset:0 atIndex:3];
        NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > total) tg = (NSUInteger)total;
        if (tg == 0) tg = 1;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        /* consumed by the routed matvec that follows on the same queue */
        if (!ds4_gpu_submit_command_buffer_async(cb, owned)) return 0;
    }
    return 1;
}

int ds4_gpu_zchain_scale_routed(
        ds4_gpu_tensor *routed, const ds4_gpu_tensor *x,
        uint32_t layer, uint32_t n_tokens) {
    if (!g_zchain_n_layer || layer >= g_zchain_n_layer) return 1;   /* not loaded => no-op */
    const uint32_t op_start = g_zchain_layer_off[layer];
    const uint32_t op_count = g_zchain_layer_off[layer + 1] - op_start;
    const uint32_t zl_k   = g_zchain_zl_k ? g_zchain_zl_k[layer] : 0;
    const uint32_t zl_off = (zl_k && g_zchain_zl_off) ? g_zchain_zl_off[layer] : 0;
    const float    zl_tr  = (zl_k && g_zchain_zl_tr) ? g_zchain_zl_tr[layer] : 0.0f;
    const uint32_t zl_din = (zl_k && g_zchain_zl_din && g_zchain_zl_din[layer])
                                ? g_zchain_zl_din[layer] : g_zchain_d_model;
    if (op_count == 0 && zl_k == 0) return 1;
    if (!routed || !x || n_tokens == 0) return 0;
    @autoreleasepool {
        id<MTLBuffer> rbuf = ds4_gpu_tensor_buffer(routed);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t vec_bytes = (uint64_t)n_tokens * g_zchain_d_model * sizeof(float);
        if (!rbuf || !xbuf ||
            ds4_gpu_tensor_bytes(routed) < vec_bytes ||
            ds4_gpu_tensor_bytes(x) < vec_bytes) {
            fprintf(stderr, "ds4: Metal zchain scale received undersized buffers\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_dsv4_zchain_scale");
        if (!pipeline) return 0;
        struct { uint32_t d_model, n_tokens, op_start, op_count, zl_k, zl_off, zl_din; float zl_tr; } args = {
            g_zchain_d_model, n_tokens, op_start, op_count, zl_k, zl_off, zl_din, zl_tr
        };
        /* power-of-two threadgroup for the tree reduce */
        NSUInteger tg = 256;
        while (tg > pipeline.maxTotalThreadsPerThreadgroup) tg >>= 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:rbuf offset:ds4_gpu_tensor_offset(routed) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:g_zchain_ops offset:0 atIndex:3];
        [enc setBuffer:g_zchain_v8 offset:0 atIndex:4];
        [enc setBuffer:(g_zchain_zlm ? g_zchain_zlm : g_zchain_v8) offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:tg * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        /* consumed by the shared-down fusion / next layer on the same queue */
        if (!ds4_gpu_submit_command_buffer_async(cb, owned)) return 0;
    }
    return 1;
}

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
} ds4_gpu_directional_steering_project_args;

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_directional_steering_project_f32");
        if (!pipeline) return 0;

        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> dbuf = ds4_gpu_tensor_buffer(directions);
        const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
        const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
        if (!xbuf || !dbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(directions) < dir_bytes) {
            fprintf(stderr, "ds4: Metal directional steering received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        while (nth > width && nth > 1u) nth >>= 1;
        if (nth == 0) nth = 1;

        ds4_gpu_directional_steering_project_args args = {
            .width = width,
            .rows = rows,
            .layer = layer,
            .n_threads = (uint32_t)nth,
            .scale = scale,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:dbuf offset:ds4_gpu_tensor_offset(directions) atIndex:2];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "directional steering")) return 0;
    }

    return 1;
}

/* type10 zl.HXP 层出口放大器: Metal 侧未实现 —— 有载荷时响亮拒绝(禁静默降级)。 */
int ds4_gpu_zchain_hxp_set(const uint16_t *m, const uint32_t *off, const uint32_t *k,
                           const uint32_t *hd, uint32_t n_layer, uint64_t total_halves) {
    (void)m; (void)off; (void)hd;
    for (uint32_t l = 0; k && l < n_layer; l++)
        if (k[l]) {
            fprintf(stderr, "ds4: zchain HXP(type10) Metal 未实现 -- aborting (no silent quality downgrade)\n");
            exit(1);
        }
    (void)total_halves;
    return 1;
}

int ds4_gpu_zchain_hxp_apply(void *hc, const void *x, uint32_t layer, uint32_t n_tokens) {
    (void)hc; (void)x; (void)layer; (void)n_tokens;
    return 1;
}
