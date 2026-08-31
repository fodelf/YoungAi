/* metal_moe_encode2.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_encode_mul_mv_id_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !mm_pipeline || !map_args || !mm_args ||
        !src0 || !src1 || !dst || !ids ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    return ds4_gpu_encode_mul_mm_id_map(cb,
                                          map_pipeline,
                                          map_args,
                                          mm_args,
                                          ids,
                                          ids_off) &&
           ds4_gpu_encode_mul_mm_id_mapped(cb,
                                             mm_pipeline,
                                             mm_args,
                                             src0,
                                             src0_off,
                                             src1,
                                             src1_off,
                                             dst,
                                             dst_off);
}

int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !map_args || !mm_args || !ids ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) return 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer,
                                         &g_moe_id_map_bytes,
                                         tpe_bytes + hids_bytes,
                                         "ds4_moe_id_map")) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:map_pipeline];
    [enc setBytes:map_args length:sizeof(*map_args) atIndex:0];
    [enc setBuffer:ids offset:ids_off atIndex:1];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:(NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne20 * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake((NSUInteger)mm_args->ne02, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_mul_mm_id_mapped_tile(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    if (!cb || !mm_pipeline || !mm_args || !src0 || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }
    /*
     * The routed MoE grouped matmul uses the legacy 32-token expert-major tile.
     * The removed TensorOps variant was not semantically stable on evals, so keep
     * this encoder tied to the tested simdgroup kernel shape.
     */
    const NSUInteger tile_n = 32u;

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc setThreadgroupMemoryLength:8192u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + tile_n - 1u) / tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    return ds4_gpu_encode_mul_mm_id_mapped_tile(cb,
                                                  mm_pipeline,
                                                  mm_args,
                                                  src0,
                                                  src0_off,
                                                  src1,
                                                  src1_off,
                                                  dst,
                                                  dst_off);
}

int ds4_gpu_encode_attn_out_low_q8_mpp(
        id<MTLCommandBuffer>           cb,
        id<MTLComputePipelineState>    pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>                  src0,
        NSUInteger                     src0_off,
        id<MTLBuffer>                  src1,
        NSUInteger                     src1_off,
        id<MTLBuffer>                  dst,
        NSUInteger                     dst_off) {
    if (!cb || !pipeline || !mm_args || !src0 || !src1 || !dst ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne02 <= 0 || mm_args->ne1 <= 0 || mm_args->ne21 <= 0) {
        return 0;
    }

    const uint32_t tile_n = DS4_METAL_ATTN_OUT_MPP_TILE_N;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst offset:dst_off atIndex:3];
    [enc setThreadgroupMemoryLength:4096u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + (NSUInteger)tile_n - 1u) / (NSUInteger)tile_n,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_swiglu_flat(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !gate || !up || !out || n == 0) return 0;

    ds4_gpu_glu_args args = {
        .ne00 = (int32_t)n,
        .nb01 = (uint64_t)n * sizeof(float),
        .ne10 = (int32_t)n,
        .nb11 = (uint64_t)n * sizeof(float),
        .ne0 = (int32_t)n,
        .nb1 = (uint64_t)n * sizeof(float),
        .i00 = 0,
        .i10 = 0,
        .alpha = 1.0f,
        .limit = 0.0f,
    };
    NSUInteger nth = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger ds4_nth = n > 1 ? (NSUInteger)n / 2u : 1u;
    if (nth > ds4_nth) nth = ds4_nth;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_swiglu_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate offset:gate_off atIndex:1];
    [enc setBuffer:up   offset:up_off   atIndex:2];
    [enc setBuffer:out  offset:out_off  atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_moe_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        mid,
        NSUInteger           mid_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows,
        float                clamp_value,
        bool                 mid_f16) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline(mid_f16 ? "kernel_dsv4_moe_swiglu_weight_f16" :
                                         "kernel_dsv4_moe_swiglu_weight");
    if (!pipeline) return 0;

    ds4_gpu_dsv4_moe_swiglu_weight_args args = {
        .width = width,
        .rows = rows,
        .gate_row_stride = (uint64_t)width * sizeof(float),
        .up_row_stride = (uint64_t)width * sizeof(float),
        .mid_row_stride = (uint64_t)width * (mid_f16 ? sizeof(uint16_t) : sizeof(float)),
        .weight_stride = sizeof(float),
        /* 恒 0: kernel 侧 uniform 保留(ABI 与 shader 对齐), 写回 clamped 激活的
         * 诊断路已删 —— 置 1 会强制非融合 kernel, 那是它删除的原因之一。 */
        .write_clamped = 0u,
        .clamp_value = clamp_value,
    };

    NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate    offset:gate_off    atIndex:1];
    [enc setBuffer:up      offset:up_off      atIndex:2];
    [enc setBuffer:mid     offset:mid_off     atIndex:3];
    [enc setBuffer:weights offset:weights_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
