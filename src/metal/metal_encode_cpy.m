/* metal_encode_cpy.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = src_col_stride,
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f16_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = (uint64_t)rows * src_row_stride,
        .nb03 = (uint64_t)rows * src_row_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = (uint64_t)rows * dst_row_stride,
        .nb3 = (uint64_t)rows * dst_row_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f16_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_encode_copy_to_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        bool                 src_is_f16,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst) return 0;
    if (n == 0) return 1;
    if (!src_is_f16) {
        return ds4_gpu_encode_cpy_f32_f16_1d(cb, src, src_off, dst, dst_off, n);
    }

    if (g_batch_cb && cb == g_batch_cb) ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:src
            sourceOffset:src_off
                toBuffer:dst
       destinationOffset:dst_off
                    size:(NSUInteger)n * sizeof(uint16_t)];
    [blit endEncoding];
    return 1;
}

int ds4_gpu_encode_fill_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             n,
        float                value) {
    if (!cb || !buf || n == 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(n, 1, 0, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth = (NSUInteger)n;
    const NSUInteger max_threads = g_unary_fill_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}
