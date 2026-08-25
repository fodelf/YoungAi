/* metal_moe_encode1.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_encode_unary_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer>               src,
        NSUInteger                  src_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        uint32_t                    width,
        uint32_t                    rows,
        int                         c4,
        float                       min,
        float                       max) {
    if (!cb || !pipeline || !src || !dst || width == 0 || rows == 0) return 0;
    if (c4 && (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, c4, 0.0f, 0.0f);
    args.min = min;
    args.max = max;

    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off) {
    if (!cb || !pipeline || !args || !a || !b || !out || args->ne0 <= 0 || args->ne1 <= 0) {
        return 0;
    }

    const NSUInteger nth = ds4_gpu_bin_threads((uint32_t)args->ne0, pipeline);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)args->ne1,
                                          (NSUInteger)args->ne2,
                                          (NSUInteger)args->ne3)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

ds4_gpu_bin_args ds4_gpu_make_bin_rowwise_scalar_args(uint32_t width, uint32_t rows) {
    const uint64_t lhs_row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rhs_row_bytes = sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = lhs_row_bytes,
        .nb02 = (uint64_t)rows * lhs_row_bytes,
        .nb03 = (uint64_t)rows * lhs_row_bytes,
        .ne10 = 1,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = (uint64_t)rows * rhs_row_bytes,
        .nb13 = (uint64_t)rows * rhs_row_bytes,
        .ne0 = (int32_t)width,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = lhs_row_bytes,
        .nb2 = (uint64_t)rows * lhs_row_bytes,
        .nb3 = (uint64_t)rows * lhs_row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

ds4_gpu_mul_mv_id_args ds4_gpu_make_mul_mv_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t nr0) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    const uint64_t src0_blocks = src0_cols / 256u;
    const uint64_t src0_block_bytes = src0_blocks ? src0_row_bytes / src0_blocks : 1u;
    return (ds4_gpu_mul_mv_id_args) {
        .nei0 = (int32_t)selected_experts,
        .nei1 = (int32_t)n_tokens,
        .nbi1 = (uint64_t)selected_experts * sizeof(int32_t),
        .ne00 = (int32_t)src0_cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = (int32_t)src0_experts,
        .nb00 = src0_block_bytes,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .ne12 = (int32_t)n_tokens,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .nb1 = (uint64_t)src0_rows * sizeof(float),
        .nr0 = (int32_t)nr0,
    };
}

ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    return (ds4_gpu_mul_mm_id_map_args) {
        .ne02 = (int32_t)src0_experts,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne21 = (int32_t)n_tokens,
        .ne20 = (int32_t)selected_experts,
        .nb21 = (uint64_t)selected_experts * sizeof(int32_t),
        .slot_lo = 0,
        .slot_hi = (int32_t)src0_experts,   /* full range: bit-identical to un-split */
    };
}

ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    return ds4_gpu_make_mul_mm_id_args_src1_size(src0_cols,
                                                   src0_rows,
                                                   src0_experts,
                                                   src0_row_bytes,
                                                   src0_expert_bytes,
                                                   src1_expert_rows,
                                                   selected_experts,
                                                   n_tokens,
                                                   sizeof(float));
}

ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * src1_elem_size;
    return (ds4_gpu_mul_mm_id_args) {
        .ne00 = (int32_t)src0_cols,
        .ne02 = (int32_t)src0_experts,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .nb03 = (uint64_t)src0_experts * src0_expert_bytes,
        .ne11 = (int32_t)src1_expert_rows,
        .nb10 = src1_elem_size,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .nb13 = (uint64_t)n_tokens * (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne20 = (int32_t)selected_experts,
        .ne21 = (int32_t)n_tokens,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .r2 = 1,
        .r3 = 1,
    };
}

uint32_t ds4_gpu_routed_mv_nr0(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q4_K:    return 2;
    case DS4_METAL_TENSOR_Q2_K:
    case DS4_METAL_TENSOR_IQ2_XXS: return 4;
    default:                       return 0;
    }
}

const char *ds4_gpu_metal_tensor_type_name(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return "iq2_xxs";
    case DS4_METAL_TENSOR_Q2_K:    return "q2_k";
    case DS4_METAL_TENSOR_Q4_K:    return "q4_k";
    case DS4_METAL_TENSOR_GO1B:    return "go1b";
    case DS4_METAL_TENSOR_GO2B:    return "go2b";
    case DS4_METAL_TENSOR_F16W:    return "f16w";
    default:                       return "unknown";
    }
}

NSUInteger ds4_gpu_routed_mv_smem(uint32_t type) {
    if (type == DS4_METAL_TENSOR_IQ2_XXS) {
        return 256u * sizeof(uint64_t) + 128u * sizeof(uint8_t);
    }
    return 0;
}

id<MTLComputePipelineState> ds4_gpu_routed_mv_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return g_moe_mul_mv_id_iq2_xxs_pipeline;
    case DS4_METAL_TENSOR_Q2_K:    return g_moe_mul_mv_id_q2_k_pipeline;
    case DS4_METAL_TENSOR_Q4_K:    return g_moe_mul_mv_id_q4_k_pipeline;
    default:                       return nil;
    }
}

id<MTLComputePipelineState> ds4_gpu_routed_mm_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f32", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f32", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
    case DS4_METAL_TENSOR_GO1B:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_go1b_f32", false);
    case DS4_METAL_TENSOR_GO2B:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_go2b_f32", false);
    case DS4_METAL_TENSOR_F16W:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_f16w_f32", false);
    default:
        return nil;
    }
}

id<MTLComputePipelineState> ds4_gpu_routed_mm_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f16", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f16", false);
    case DS4_METAL_TENSOR_GO1B:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_go1b_f16", false);
    case DS4_METAL_TENSOR_GO2B:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_go2b_f16", false);
    case DS4_METAL_TENSOR_F16W:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_f16w_f16", false);
    default:
        return nil;
    }
}

int ds4_gpu_encode_mul_mv_id(
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
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

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
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_mul_mv_id_pair(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0_a offset:src0_a_off atIndex:1];
    [enc setBuffer:src0_b offset:src0_b_off atIndex:2];
    [enc setBuffer:src1   offset:src1_off   atIndex:3];
    [enc setBuffer:dst_a  offset:dst_a_off  atIndex:4];
    [enc setBuffer:dst_b  offset:dst_b_off  atIndex:5];
    [enc setBuffer:ids    offset:ids_off    atIndex:6];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_encode_mul_mv_id_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act ||
        !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:src0_a  offset:src0_a_off  atIndex:2];
    [enc setBuffer:src0_b  offset:src0_b_off  atIndex:3];
    [enc setBuffer:src1    offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids     offset:ids_off     atIndex:8];
    [enc setBuffer:weights offset:weights_off atIndex:9];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}
