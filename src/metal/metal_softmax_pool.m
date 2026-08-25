/* metal_softmax_pool.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

typedef struct {
    uint32_t width;
} ds4_gpu_dsv4_ratio4_shift_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_softmax_pool_args;

static ds4_gpu_bin_args ds4_gpu_make_bin_contiguous_3d_args(
        uint32_t cols,
        uint32_t rows,
        uint32_t planes) {
    const uint64_t row_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)rows,
        .ne12 = (int32_t)planes,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = (int32_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_softmax_f32_contiguous(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows,
        uint32_t             planes) {
    if (!cb || !src || !dst || width == 0 || rows == 0 || planes == 0) return 0;

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    ds4_gpu_softmax_args args = {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne11 = (int32_t)width,
        .ne12 = (int32_t)rows,
        .ne13 = (int32_t)planes,
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .scale = 1.0f,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 1,
    };

    id<MTLComputePipelineState> pipeline =
        (width % 4u) == 0 ? g_soft_max_f32_4_pipeline : g_soft_max_f32_pipeline;
    if (!pipeline) return 0;

    NSUInteger nth = 32u;
    if ((width % 4u) == 0) {
        while (nth < (NSUInteger)(width / 4u) &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    } else {
        while (nth < (NSUInteger)width &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    }
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:src offset:src_off atIndex:2];
    [enc setBuffer:src offset:src_off atIndex:3];
    [enc setBuffer:dst offset:dst_off atIndex:4];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf || n_rows == 0 || head_dim == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * sizeof(float)) {
        return 0;
    }

    const NSUInteger pack_bytes = (NSUInteger)n_rows * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_product_buffer,
                                         &g_compressor_pool_product_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_product") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_cont_buffer,
                                         &g_compressor_pool_score_cont_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_score_cont") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_softmax_buffer,
                                         &g_compressor_pool_softmax_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_softmax")) {
        return 0;
    }

    const uint64_t cont_row_stride = (uint64_t)n_rows * sizeof(float);
    const uint64_t cont_plane_stride = (uint64_t)head_dim * cont_row_stride;

    /*
     * Keep the n_comp == 1 compressor path as the unfused graph sequence:
     *
     *   score = soft_max(contiguous(score))
     *   pooled = sum_rows(contiguous(kv) * score)
     *
     * The fused DS4 pool kernel is mathematically equivalent, but it reduces in
     * a different order. That is enough to create ~1e-6 compressor differences
     * and later FP8/routing flips, so this path intentionally keeps the same
     * operation boundary and memory layout as the graph.
     */
    ds4_gpu_bin_args mul_args =
        ds4_gpu_make_bin_contiguous_3d_args(n_rows, head_dim, 1);

    return
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    kvbuf,
                                                    kv_offset,
                                                    g_compressor_pool_product_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    kv_nb0,
                                                    kv_nb1,
                                                    kv_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    scorebuf,
                                                    score_offset,
                                                    g_compressor_pool_score_cont_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    score_nb0,
                                                    score_nb1,
                                                    score_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_softmax_f32_contiguous(cb,
                                                g_compressor_pool_score_cont_buffer,
                                                0,
                                                g_compressor_pool_softmax_buffer,
                                                0,
                                                n_rows,
                                                head_dim,
                                                1) &&
        ds4_gpu_encode_bin_f32_rows(cb,
                                      g_mul_pipeline,
                                      &mul_args,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      g_compressor_pool_softmax_buffer,
                                      0,
                                      g_compressor_pool_product_buffer,
                                      0) &&
        ds4_gpu_encode_sum_rows_f32(cb,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      outbuf,
                                      ds4_gpu_tensor_offset(out),
                                      n_rows,
                                      head_dim);
}

int ds4_gpu_encode_dsv4_softmax_pool(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim,
        uint32_t             n_comp) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf ||
        n_rows == 0 || head_dim == 0 || n_comp == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * n_comp * sizeof(float)) {
        return 0;
    }

    if (n_comp == 1) {
        return ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(cb,
                                                                out,
                                                                kvbuf,
                                                                kv_offset,
                                                                kv_nb0,
                                                                kv_nb1,
                                                                kv_nb2,
                                                                scorebuf,
                                                                score_offset,
                                                                score_nb0,
                                                                score_nb1,
                                                                score_nb2,
                                                                n_rows,
                                                                head_dim);
    }

    ds4_gpu_dsv4_softmax_pool_args args = {
        .ne00 = (int64_t)n_rows,
        .ne01 = (int64_t)head_dim,
        .ne02 = (int64_t)n_comp,
        .nb00 = kv_nb0,
        .nb01 = kv_nb1,
        .nb02 = kv_nb2,
        .nb10 = score_nb0,
        .nb11 = score_nb1,
        .nb12 = score_nb2,
        .ne0 = (int64_t)head_dim,
        .ne1 = (int64_t)n_comp,
        .nb0 = sizeof(float),
        .nb1 = (uint64_t)head_dim * sizeof(float),
    };
    const uint64_t n = (uint64_t)head_dim * n_comp;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_softmax_pool_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:kv_offset atIndex:1];
    [enc setBuffer:scorebuf offset:score_offset atIndex:2];
    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_concat_f32_dim1(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src0,
        NSUInteger           src0_offset,
        uint32_t             src0_rows,
        uint64_t             src0_row_stride,
        id<MTLBuffer>        src1,
        NSUInteger           src1_offset,
        uint32_t             src1_rows,
        uint64_t             src1_row_stride,
        id<MTLBuffer>        dst,
        NSUInteger           dst_offset,
        uint32_t             cols,
        uint64_t             dst_row_stride) {
    if (!cb || !src0 || !src1 || !dst || cols == 0 || src0_rows == 0 || src1_rows == 0) {
        return 0;
    }

    const uint32_t rows = src0_rows + src1_rows;
    const uint64_t src0_plane = (uint64_t)src0_rows * src0_row_stride;
    const uint64_t src1_plane = (uint64_t)src1_rows * src1_row_stride;
    const uint64_t dst_plane = (uint64_t)rows * dst_row_stride;
    ds4_gpu_concat_args args = {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_row_stride,
        .nb02 = src0_plane,
        .nb03 = src0_plane,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)src1_rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_stride,
        .nb12 = src1_plane,
        .nb13 = src1_plane,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane,
        .nb3 = dst_plane,
        .dim = 1,
    };

    NSUInteger nth = cols < 1024u ? (NSUInteger)cols : 1024u;
    const NSUInteger max_threads = g_concat_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_concat_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src0 offset:src0_offset atIndex:1];
    [enc setBuffer:src1 offset:src1_offset atIndex:2];
    [enc setBuffer:dst offset:dst_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_compressor_pool(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *out,
        const ds4_gpu_tensor *state_kv,
        const ds4_gpu_tensor *state_score,
        uint32_t               head_dim,
        uint32_t               ratio) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !out || !statekvbuf || !statescbuf || head_dim == 0 || ratio == 0) return 0;

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rows = coff * ratio;
    const uint64_t state_bytes = (uint64_t)width * rows * sizeof(float);
    if (ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    if (ratio != 4u) {
        const uint64_t row_stride = (uint64_t)width * sizeof(float);
        return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                  out,
                                                  statekvbuf,
                                                  ds4_gpu_tensor_offset(state_kv),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  statescbuf,
                                                  ds4_gpu_tensor_offset(state_score),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  ratio,
                                                  head_dim,
                                                  1);
    }

    const NSUInteger packed_bytes = (NSUInteger)8u * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                         &g_compressor_pool_kv_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                         &g_compressor_pool_score_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_score")) {
        return 0;
    }

    const uint64_t state_row_stride = (uint64_t)width * sizeof(float);
    const uint64_t pool_row_stride = (uint64_t)head_dim * sizeof(float);
    const NSUInteger curr_offset = (NSUInteger)4u * state_row_stride +
                                   (NSUInteger)head_dim * sizeof(float);
    if (!ds4_gpu_encode_concat_f32_dim1(cb,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv),
                                          4,
                                          state_row_stride,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_kv_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride) ||
        !ds4_gpu_encode_concat_f32_dim1(cb,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score),
                                          4,
                                          state_row_stride,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_score_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride)) {
        return 0;
    }

    return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                              out,
                                              g_compressor_pool_kv_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              g_compressor_pool_score_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              8,
                                              head_dim,
                                              1);
}

int ds4_gpu_encode_compressor_shift_ratio4(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        uint32_t             width) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !statekvbuf || !statescbuf || !g_dsv4_ratio4_shift_pipeline || width == 0) return 0;

    ds4_gpu_dsv4_ratio4_shift_args args = { .width = width };
    const uint32_t n = 4u * width;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_ratio4_shift_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:1];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
