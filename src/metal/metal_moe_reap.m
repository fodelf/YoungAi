/* metal_moe_reap.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_kargs_sum_rows;

typedef struct {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
} ds4_gpu_dsv4_moe_sum6_args;

typedef struct { uint32_t tokens, width, n_used, layer; uint64_t src_token_stride; } ds4_gpu_reap_args;

static int           g_reap_enabled = -1;

static id<MTLBuffer> g_reap_saliency_buf;

/* [DS4_REAP_MAX_LAYERS*256] uint32 fixed-point */
static id<MTLBuffer> g_reap_counts_buf;

/* [DS4_REAP_MAX_LAYERS*256] uint32 */
static id<MTLComputePipelineState> g_reap_accum_pipeline;

/* stashed at router-select, consumed at the immediately-following sum6 (same layer,
 * in-order cb execution makes the per-layer pairing exact even with a reused buffer) */
id<MTLBuffer> g_reap_sel_buf;

NSUInteger    g_reap_sel_off;

uint32_t      g_reap_layer;

uint32_t      g_reap_n_used;

static void ds4_gpu_reap_dump(void) {
    if (g_reap_enabled != 1 || !g_reap_saliency_buf || !g_reap_counts_buf) return;
    const uint32_t *sal = (const uint32_t *)g_reap_saliency_buf.contents;
    const uint32_t *cnt = (const uint32_t *)g_reap_counts_buf.contents;
    /* Write to a BUFFERED file, not unbuffered stderr: a SIGTERM'd worker has only a
     * brief window before the harness escalates to SIGKILL, and ~5k line-buffered
     * stderr syscalls lose the race.  layer = per-machine first-seen slot (coord slot s
     * == global layer s; worker slot s == global layer 20+s -- remap when merging). */
    const char *path = getenv("DS4_REAP_DUMP_FILE");
    FILE *f = fopen(path && *path ? path : "/tmp/reap_saliency.txt", "w");
    if (!f) { fprintf(stderr, "ds4: REAP dump fopen failed\n"); return; }
    for (uint32_t L = 0; L < DS4_REAP_MAX_LAYERS; L++) {
        uint32_t used = 0;
        for (uint32_t e = 0; e < 256u; e++) if (cnt[L * 256u + e]) used++;
        if (!used) continue;
        bool printed[256] = { false };
        for (uint32_t r = 0; r < 256u; r++) {
            int best = -1; double bestS = -1.0;
            for (uint32_t e = 0; e < 256u; e++) {
                if (printed[e] || !cnt[L * 256u + e]) continue;
                const double S = (double)sal[L * 256u + e] / (256.0 * (double)cnt[L * 256u + e]);
                if (S > bestS) { bestS = S; best = (int)e; }
            }
            if (best < 0) break;
            printed[best] = true;
            fprintf(f, "reap-saliency L%02u E%03d S=%.5f count=%u\n",
                    L, best, bestS, cnt[L * 256u + (uint32_t)best]);
        }
    }
    fclose(f);
    fprintf(stderr, "ds4: REAP saliency written to %s\n", path && *path ? path : "/tmp/reap_saliency.txt");
}

/* A layer-sliced worker is shut down with SIGTERM (no atexit), so its 20-42 saliency
 * would never dump.  When REAP is collecting, route SIGTERM through exit() so the
 * atexit(reap_dump) fires -- by then its last forward's cb has completed and the
 * Shared accumulator buffers are populated. */
static void ds4_gpu_reap_sigterm(int sig) { (void)sig; exit(0); }

int ds4_gpu_reap_enabled(void) {
    if (g_reap_enabled < 0) {
        g_reap_enabled = ds4_gpu_env_bool("DS4_REAP_COLLECT") > 0 ? 1 : 0;
        if (g_reap_enabled == 1) {
            const size_t n = (size_t)DS4_REAP_MAX_LAYERS * 256u * sizeof(uint32_t);
            g_reap_saliency_buf = [g_device newBufferWithLength:n options:MTLResourceStorageModeShared];
            g_reap_counts_buf   = [g_device newBufferWithLength:n options:MTLResourceStorageModeShared];
            g_reap_accum_pipeline = ds4_gpu_get_pipeline("kernel_dsv4_reap_accum");
            if (!g_reap_saliency_buf || !g_reap_counts_buf || !g_reap_accum_pipeline) {
                fprintf(stderr, "ds4: REAP collect init failed\n");
                g_reap_enabled = 0;
            } else {
                memset(g_reap_saliency_buf.contents, 0, n);
                memset(g_reap_counts_buf.contents, 0, n);
                atexit(ds4_gpu_reap_dump);
                signal(SIGTERM, ds4_gpu_reap_sigterm);
                fprintf(stderr, "ds4: REAP saliency collection enabled (DS4_REAP_COLLECT=1)\n");
            }
        }
    }
    return g_reap_enabled;
}

/* Dispatch reap_accum over the sum6 input (the per-expert gated outputs). */
static void ds4_gpu_reap_dispatch(id<MTLCommandBuffer> cb, id<MTLBuffer> experts,
                                  NSUInteger experts_off, uint32_t width, uint32_t n_tokens) {
    if (ds4_gpu_reap_enabled() != 1 || !cb || !experts || !g_reap_sel_buf ||
        !g_reap_accum_pipeline || width == 0 || n_tokens == 0) return;
    const uint32_t n_used = g_reap_n_used ? g_reap_n_used : 6u;
    ds4_gpu_reap_args args = { .tokens = n_tokens, .width = width, .n_used = n_used,
                               .layer = g_reap_layer,
                               .src_token_stride = (uint64_t)n_used * width * sizeof(float) };
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_reap_accum_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:g_reap_sel_buf offset:g_reap_sel_off atIndex:2];
    [enc setBuffer:g_reap_saliency_buf offset:0 atIndex:3];
    [enc setBuffer:g_reap_counts_buf offset:0 atIndex:4];
    const NSUInteger total = (NSUInteger)n_tokens * n_used;
    NSUInteger tg = g_reap_accum_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u; if (tg == 0u) tg = 1u;
    const NSUInteger groups = (total + tg - 1u) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
}

static int ds4_gpu_encode_moe_sum6(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_tokens == 0) return 0;

    if (!g_moe_sum6_pipeline) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_dsv4_moe_sum6_args args = {
        .width = out_dim,
        .tokens = n_tokens,
        .src_token_stride = 6u * out_row_bytes,
        .dst_token_stride = out_row_bytes,
    };

    NSUInteger nth = g_moe_sum6_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > out_dim) nth = out_dim;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_sum6_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:experts offset:experts_off atIndex:1];
    [enc setBuffer:out     offset:out_off     atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    /* REAP: ||gated expert output||_2 per (token,slot) -> saliency, keyed by the
     * router selection stashed at this layer's router-select (no-op unless enabled). */
    ds4_gpu_reap_dispatch(cb, experts, experts_off, out_dim, n_tokens);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

int ds4_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    if (n_expert == 6 &&
        ds4_gpu_encode_moe_sum6(cb,
                                  experts,
                                  experts_off,
                                  out,
                                  out_off,
                                  out_dim,
                                  n_tokens)) {
        return 1;
    }

    ds4_gpu_bin_args first =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!ds4_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    ds4_gpu_bin_args accum =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!ds4_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_encode_get_rows_i32_token_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        table,
        NSUInteger           table_off,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_off,
        const int32_t       *token_inline,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        uint32_t             hash_rows,
        uint32_t             n_cols,
        uint32_t             n_tokens) {
    if (!cb || !table || !selected || hash_rows == 0 || n_cols == 0 || n_tokens == 0) return 0;
    if (!tokens && !token_inline) return 0;

    const uint64_t table_row_bytes = (uint64_t)n_cols * sizeof(int32_t);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int64_t)n_cols,
        .ne00 = (int64_t)n_cols,
        .nb01 = table_row_bytes,
        .nb02 = (uint64_t)hash_rows * table_row_bytes,
        .nb03 = (uint64_t)hash_rows * table_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = table_row_bytes,
        .nb2 = (uint64_t)n_tokens * table_row_bytes,
        .nb3 = (uint64_t)n_tokens * table_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_cols;
    const NSUInteger max_threads = g_get_rows_i32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;
    const NSUInteger nw0 = ((NSUInteger)n_cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:table offset:table_off atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_off atIndex:2];
    } else {
        [enc setBytes:token_inline length:sizeof(*token_inline) atIndex:2];
    }
    [enc setBuffer:selected offset:selected_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_get_rows_f32_router_weights(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        probs,
        NSUInteger           probs_off,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             n_expert,
        uint32_t             n_expert_used,
        uint32_t             n_tokens) {
    if (!cb || !probs || !selected || !weights || n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;

    const uint64_t probs_token_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_row_bytes = (uint64_t)n_expert_used * sizeof(int32_t);
    const uint64_t weights_row_bytes = (uint64_t)n_expert_used * sizeof(float);
    ds4_gpu_get_rows_args args = {
        .ne00t = 1,
        .ne00 = 1,
        .nb01 = sizeof(float),
        .nb02 = probs_token_bytes,
        .nb03 = (uint64_t)n_tokens * probs_token_bytes,
        .ne10 = (int64_t)n_expert_used,
        .nb10 = sizeof(int32_t),
        .nb11 = selected_row_bytes,
        .nb12 = (uint64_t)n_tokens * selected_row_bytes,
        .nb1 = sizeof(float),
        .nb2 = weights_row_bytes,
        .nb3 = (uint64_t)n_tokens * weights_row_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:probs offset:probs_off atIndex:1];
    [enc setBuffer:selected offset:selected_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_expert_used, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || width == 0 || rows == 0) return 0;

    const uint64_t src_row_bytes = (uint64_t)width * sizeof(float);
    ds4_gpu_kargs_sum_rows args = {
        .ne00 = (int64_t)width,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)rows * src_row_bytes,
        .nb03 = (uint64_t)rows * src_row_bytes,
        .ne0 = 1,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = sizeof(float),
        .nb2 = (uint64_t)rows * sizeof(float),
        .nb3 = (uint64_t)rows * sizeof(float),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_sum_rows_f32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)args.ne00 && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)args.ne00) nth = (NSUInteger)args.ne00;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_sum_rows_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

/* REAP expert keep-mask (Stage 4a quality test): restrict the router to the kept
 * experts (top-K saliency) per layer by reusing the bias path -- pruned experts get
 * bias -inf so they never enter the top-6, and the router renormalizes over the
 * survivors.  No repack / no memory change here -- this checks whether the pruned
 * model holds Go+React quality before committing to a repack.  Keep-set is
 * slot-indexed (machine-local first-seen layer order, matching the saliency dump). */
bool g_reap_keep[DS4_ROUTER_CACHE_HOT_LAYERS][256];
