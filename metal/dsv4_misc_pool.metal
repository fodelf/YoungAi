kernel void kernel_dsv4_indexer_weighted_sum(
        constant ds4_metal_args_dsv4_indexer_weighted_sum & args,
        device const char * scores,
        device const char * weights,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((device const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((device const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += max(s, 0.0f) * (w * args.scale);
    }

    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

// Fused softmax-weighted pooling of compressed KV rows. It is used when several
// compressor rows are present; the one-row case deliberately follows the
// unfused softmax/mul/sum graph in Objective-C to keep identical reductions.
kernel void kernel_dsv4_softmax_pool(
        constant ds4_metal_args_dsv4_softmax_pool & args,
        device const char * kv,
        device const char * score,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = exp(s - max_s);
        const float v = *((device const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((device float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}
