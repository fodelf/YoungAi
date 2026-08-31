/* metal_corr.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (!isfinite(clamp) || clamp < 0.0f || !isfinite(weight)) return 0;

    @autoreleasepool {
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!gatebuf || !upbuf || !outbuf ||
            ds4_gpu_tensor_bytes(gate) < bytes ||
            ds4_gpu_tensor_bytes(up) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal SwiGLU received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glu_args args = {
            .ne00 = (int32_t)n,
            .nb01 = (uint64_t)n * sizeof(float),
            .ne10 = (int32_t)n,
            .nb11 = (uint64_t)n * sizeof(float),
            .ne0 = (int32_t)n,
            .nb1 = (uint64_t)n * sizeof(float),
            .i00 = 0,
            .i10 = 0,
            .alpha = weight,
            .limit = clamp,
        };
        NSUInteger nth = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger ds4_nth = n > 1 ? (NSUInteger)n / 2u : 1u;
        if (nth > ds4_nth) nth = ds4_nth;
        if (nth == 0) nth = 1;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "SwiGLU")) return 0;
    }

    return 1;
}

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = ds4_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !outbuf ||
            ds4_gpu_tensor_bytes(a) < bytes ||
            ds4_gpu_tensor_bytes(b) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal tensor add received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        ds4_gpu_bin_args args = {
            .ne00 = (int32_t)n,
            .ne01 = 1,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = row_bytes,
            .nb02 = row_bytes,
            .nb03 = row_bytes,
            .ne10 = (int32_t)n,
            .ne11 = 1,
            .ne12 = 1,
            .ne13 = 1,
            .nb10 = sizeof(float),
            .nb11 = row_bytes,
            .nb12 = row_bytes,
            .nb13 = row_bytes,
            .ne0 = (int32_t)n,
            .ne1 = 1,
            .ne2 = 1,
            .ne3 = 1,
            .nb0 = sizeof(float),
            .nb1 = row_bytes,
            .nb2 = row_bytes,
            .nb3 = row_bytes,
            .offs = 0,
            .o1 = { 0 },
        };
        NSUInteger nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth_max > 256u) nth_max = 256u;
        NSUInteger nth = 1;
        while (2u * nth < (NSUInteger)args.ne0 && nth < nth_max) {
            nth *= 2u;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:ds4_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:ds4_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "tensor add")) return 0;
    }

    return 1;
}

/* go1b correction: add the per-expert router-logit bias delta[e] to the raw
 * router logits before top-k selection, broadcast across all n_tokens rows. */
int ds4_gpu_corr_router_bias(
        ds4_gpu_tensor       *logits,
        const ds4_gpu_tensor *delta,
        uint32_t                n_expert,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!logits || !delta || n_expert == 0 || n_tokens == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> lbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> dbuf = ds4_gpu_tensor_buffer(delta);
        const uint64_t total = (uint64_t)n_tokens * n_expert;
        if (!lbuf || !dbuf ||
            ds4_gpu_tensor_bytes(logits) < total * sizeof(float) ||
            ds4_gpu_tensor_bytes(delta) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal corr router bias received undersized buffers\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_corr_router_bias");
        if (!pipeline) return 0;

        struct { uint32_t n_expert; uint32_t n_tokens; } args = { n_expert, n_tokens };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:lbuf offset:ds4_gpu_tensor_offset(logits) atIndex:1];
        [enc setBuffer:dbuf offset:ds4_gpu_tensor_offset(delta) atIndex:2];
        NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > total) tg = (NSUInteger)total;
        if (tg == 0) tg = 1;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        /* consumed by the router select that follows on the same queue —
         * async submit, no host wait (resident buffers only) */
        if (!ds4_gpu_submit_command_buffer_async(cb, owned)) return 0;
    }
    return 1;
}

/* go1b correction: out[t][d] += sum over selected e of
 *   ( U @ ( C[e] (.*) (V @ x[t]) ) )[d] + b[d] + beta[e].
 * One threadgroup per token; vx (=V@x[t]) lives in threadgroup memory.
 * kernel_name selects in-place accumulate ("kernel_dsv4_corr_apply") vs the
 * store-to-delta variant ("kernel_dsv4_corr_delta") — same math, the delta
 * form exists so decode can keep the hot routed_out free of a tiny-dispatch
 * write hazard (measured ~23ms/layer pipeline bubble). */
static int ds4_gpu_corr_apply_impl(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t                d_model,
        uint32_t                d_l,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        uint32_t                n_tokens,
        const char             *kernel_name) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !U || !V || !C || !b || !beta || !selected ||
        d_model == 0 || d_l == 0 || n_expert == 0 || n_expert_used == 0 || n_tokens == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> xbuf   = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> ubuf   = ds4_gpu_tensor_buffer(U);
        id<MTLBuffer> vbuf   = ds4_gpu_tensor_buffer(V);
        id<MTLBuffer> cbuf   = ds4_gpu_tensor_buffer(C);
        id<MTLBuffer> bbuf   = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> betabuf = ds4_gpu_tensor_buffer(beta);
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        const uint64_t vec_bytes = (uint64_t)n_tokens * d_model * sizeof(float);
        if (!outbuf || !xbuf || !ubuf || !vbuf || !cbuf || !bbuf || !betabuf || !selbuf ||
            ds4_gpu_tensor_bytes(out) < vec_bytes ||
            ds4_gpu_tensor_bytes(x) < vec_bytes ||
            ds4_gpu_tensor_bytes(U) < (uint64_t)d_model * d_l * sizeof(float) ||
            ds4_gpu_tensor_bytes(V) < (uint64_t)d_l * d_model * sizeof(float) ||
            ds4_gpu_tensor_bytes(C) < (uint64_t)n_expert * d_l * sizeof(float) ||
            ds4_gpu_tensor_bytes(b) < (uint64_t)d_model * sizeof(float) ||
            ds4_gpu_tensor_bytes(beta) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal corr apply received undersized buffers\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline(kernel_name);
        if (!pipeline) return 0;

        struct {
            uint32_t d_model;
            uint32_t d_l;
            uint32_t n_expert;
            uint32_t n_expert_used;
            uint32_t n_tokens;
        } args = { d_model, d_l, n_expert, n_expert_used, n_tokens };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:outbuf  offset:ds4_gpu_tensor_offset(out) atIndex:1];
        [enc setBuffer:xbuf    offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:ubuf    offset:ds4_gpu_tensor_offset(U) atIndex:3];
        [enc setBuffer:vbuf    offset:ds4_gpu_tensor_offset(V) atIndex:4];
        [enc setBuffer:cbuf    offset:ds4_gpu_tensor_offset(C) atIndex:5];
        [enc setBuffer:bbuf    offset:ds4_gpu_tensor_offset(b) atIndex:6];
        [enc setBuffer:betabuf offset:ds4_gpu_tensor_offset(beta) atIndex:7];
        [enc setBuffer:selbuf  offset:ds4_gpu_tensor_offset(selected) atIndex:8];
        [enc setThreadgroupMemoryLength:(NSUInteger)d_l * sizeof(float) atIndex:0];
        NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tg > 256u) tg = 256u;
        if (tg == 0) tg = 1;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        /* routed_out is consumed by the shared-expert add / next-layer work on
         * the same queue; CPU consumers sit behind pending-drain — async submit
         * removes the per-layer commit+waitUntilCompleted pair (resident
         * buffers only, no transients) */
        if (!ds4_gpu_submit_command_buffer_async(cb, owned)) return 0;
    }
    return 1;
}

int ds4_gpu_corr_apply(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t d_model, uint32_t d_l, uint32_t n_expert,
        uint32_t n_expert_used, uint32_t n_tokens) {
    return ds4_gpu_corr_apply_impl(out, x, U, V, C, b, beta, selected,
                                   d_model, d_l, n_expert, n_expert_used,
                                   n_tokens, "kernel_dsv4_corr_apply");
}

int ds4_gpu_corr_apply_delta(
        ds4_gpu_tensor       *delta_out,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U,
        const ds4_gpu_tensor *V,
        const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *selected,
        uint32_t d_model, uint32_t d_l, uint32_t n_expert,
        uint32_t n_expert_used, uint32_t n_tokens) {
    return ds4_gpu_corr_apply_impl(delta_out, x, U, V, C, b, beta, selected,
                                   d_model, d_l, n_expert, n_expert_used,
                                   n_tokens, "kernel_dsv4_corr_delta");
}

int ds4_gpu_corr_delta_supported(void) { return 1; }
