/* metal_init.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
/* EXCEPTION(>500行): 单函数 ds4_gpu_init, 函数内拆分是后续工序 */
#import "metal_internal.h"

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small ds4_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants. */
int ds4_gpu_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "ds4: Metal device not available\n");
            return 0;
        }
        ds4_gpu_print_device_summary();
        ds4_gpu_detect_metal4_features();

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "ds4: failed to create Metal command queue\n");
            g_device = nil;
            return 0;
        }
        g_model_buffer_cache = [NSMutableDictionary dictionary];
        g_pipeline_cache = [NSMutableDictionary dictionary];
        g_transient_buffers = [NSMutableArray array];
        g_pending_cbs = [NSMutableArray array];
        if (!g_model_buffer_cache || !g_pipeline_cache || !g_transient_buffers || !g_pending_cbs) {
            fprintf(stderr, "ds4: Metal bookkeeping allocation failed\n");
            g_pending_cbs = nil;
            g_transient_buffers = nil;
            g_pipeline_cache = nil;
            g_model_buffer_cache = nil;
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        NSError *error = nil;
        NSString *source = ds4_gpu_full_source();
        if (!source) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        NSMutableDictionary *macros = [NSMutableDictionary new];
        if (g_metal4_tensor_api_enabled) {
            macros[@"DS4_METAL_HAS_TENSOR"] = @"1";
            fprintf(stderr, "ds4: Metal 4 tensor API enabled for Tensor kernels\n");
        }

        const int drift_hc_stable        = ds4_gpu_env_bool("DS4_METAL_HC_STABLE")          != 0; // default ON
        const int drift_norm_unify       = ds4_gpu_env_bool("DS4_METAL_NORM_RSQRT_DISABLE") != 0; // default ON
        const int drift_kv_raw_f32       = ds4_gpu_env_bool("DS4_METAL_KV_RAW_F32")         >  0; // default OFF
        const int drift_rope_exp2_log2   = ds4_gpu_env_bool("DS4_METAL_ROPE_EXP2_LOG2")     >  0; // default OFF
        const int drift_math_safe        = ds4_gpu_env_bool("DS4_METAL_MATH_SAFE")          >  0; // default OFF

        if (drift_math_safe) {
            // MTLCompileOptions.fastMathEnabled defaults to YES and Apple's
            // headers explicitly say this "may violate the IEEE 754 standard".
            // Different fast-math optimizations get applied across the
            // matmul2d cooperative-tensor path and the legacy
            // simdgroup_multiply_accumulate path on M5, amplifying the
            // mismatch. MTLMathModeSafe pins the entire library to strict
            // IEEE-754 semantics. Diagnostic-only: useful to localize drift
            // sources but not to ship as a default.
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                fprintf(stderr, "ds4: Metal shader library math mode = safe (strict IEEE-754) by DS4_METAL_MATH_SAFE\n");
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                fprintf(stderr, "ds4: Metal shader library fast-math disabled by DS4_METAL_MATH_SAFE (pre-macOS 15)\n");
            }
        }

        if (drift_hc_stable)      macros[@"DS4_METAL_HC_STABLE"]          = @"1";
        if (drift_norm_unify)     macros[@"DS4_METAL_NORM_RSQRT_DISABLE"] = @"1";
        if (drift_kv_raw_f32)     macros[@"DS4_METAL_KV_RAW_F32"]         = @"1";
        if (drift_rope_exp2_log2) macros[@"DS4_METAL_ROPE_EXP2_LOG2"]     = @"1";
        fprintf(stderr,
                "ds4: drift-patch flags hc_stable=%s norm_unify=%s kv_raw_f32=%s rope_exp2_log2=%s math_safe=%s tensor_matmul=%s\n",
                drift_hc_stable      ? "on"  : "off",
                drift_norm_unify     ? "on"  : "off",
                drift_kv_raw_f32     ? "on"  : "off",
                drift_rope_exp2_log2 ? "on"  : "off",
                drift_math_safe      ? "on"  : "off",
                g_metal4_tensor_api_enabled ? "on" : "off");
        options.preprocessorMacros = macros;
        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "ds4: Metal shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_library = library;

        id<MTLFunction> fn = [library newFunctionWithName:@"kernel_get_rows_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_repeat_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_repeat_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_repeat_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_set_rows_f32_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_set_rows_f32_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_set_rows_f32_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_concat"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_concat function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_concat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_concat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_concat pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f16_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f16_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_fp8_kv_quantize_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_fp8_kv_quantize_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_fp8_kv_quantize_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_hadamard_fp4_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_qat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_qat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_hadamard_fp4_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_kv_fp8_store_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_kv_fp8_store_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_kv_fp8_store_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_ratio4_shift_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_ratio4_shift_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_ratio4_shift_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_moe_sum6_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_moe_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_moe_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_op = 0;
        int16_t bin_f = 1;
        bool bin_rb = false;
        bool bin_cb = false;
        [bin_constants setConstantValue:&bin_op type:MTLDataTypeShort atIndex:1300];
        [bin_constants setConstantValue:&bin_f  type:MTLDataTypeShort atIndex:1301];
        [bin_constants setConstantValue:&bin_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_constants setConstantValue:&bin_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_add_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_plain_op = 2;
        int16_t bin_mul_plain_f = 1;
        bool bin_mul_plain_rb = false;
        bool bin_mul_plain_cb = false;
        [bin_mul_constants setConstantValue:&bin_mul_plain_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_constants setConstantValue:&bin_mul_plain_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_constants setConstantValue:&bin_mul_plain_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_constants setConstantValue:&bin_mul_plain_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_mul_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_scalar_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_op = 2;
        int16_t bin_mul_f = 1;
        bool bin_mul_rb = false;
        bool bin_mul_cb = true;
        [bin_mul_scalar_constants setConstantValue:&bin_mul_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_scalar_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_mul_scalar_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_mul_scalar_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_div_row_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_div_op = 3;
        int16_t bin_div_f = 1;
        bool bin_div_rb = false;
        bool bin_div_cb = true;
        [bin_div_row_constants setConstantValue:&bin_div_op type:MTLDataTypeShort atIndex:1300];
        [bin_div_row_constants setConstantValue:&bin_div_f  type:MTLDataTypeShort atIndex:1301];
        [bin_div_row_constants setConstantValue:&bin_div_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_div_row_constants setConstantValue:&bin_div_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_div_row_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_div_row_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_div_row_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_plain_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_plain_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_qkv_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_qkv_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_qkv_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t moe_mv_id_nsg = 2;
        [moe_mv_id_constants setConstantValue:&moe_mv_id_nsg type:MTLDataTypeShort atIndex:600];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rope_tail_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_batch_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_softmax_pool"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_softmax_pool_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_softmax_pool_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_4_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_4_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_merge_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_merge_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_merge_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *sum_rows_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t sum_rows_op = 10;
        [sum_rows_constants setConstantValue:&sum_rows_op type:MTLDataTypeShort atIndex:1400];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                           constantValues:sum_rows_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_sum_rows_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_sum_rows_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask_scatter"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_scatter_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_scatter_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_sinkhorn"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_sinkhorn_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_sinkhorn_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum_norm4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sigmoid_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sigmoid_op = 102;
        bool unary_cnt = false;
        [unary_sigmoid_constants setConstantValue:&unary_sigmoid_op type:MTLDataTypeShort atIndex:1200];
        [unary_sigmoid_constants setConstantValue:&unary_cnt        type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sigmoid_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sigmoid_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sigmoid_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_silu_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_silu_op = 106;
        [unary_silu_constants setConstantValue:&unary_silu_op type:MTLDataTypeShort atIndex:1200];
        [unary_silu_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_silu_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_silu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_softplus_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_softplus_op = 115;
        [unary_softplus_constants setConstantValue:&unary_softplus_op type:MTLDataTypeShort atIndex:1200];
        [unary_softplus_constants setConstantValue:&unary_cnt         type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_softplus_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_softplus_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_softplus_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sqrt_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sqrt_op = 14;
        [unary_sqrt_constants setConstantValue:&unary_sqrt_op type:MTLDataTypeShort atIndex:1200];
        [unary_sqrt_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sqrt_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sqrt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sqrt_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_clamp_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_clamp_op = 12;
        [unary_clamp_constants setConstantValue:&unary_clamp_op type:MTLDataTypeShort atIndex:1200];
        [unary_clamp_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32"
                           constantValues:unary_clamp_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_clamp_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_clamp_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_scale_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_scale_op = 10;
        [unary_scale_constants setConstantValue:&unary_scale_op type:MTLDataTypeShort atIndex:1200];
        [unary_scale_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_scale_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_scale_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_fill_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_fill_op = 11;
        [unary_fill_constants setConstantValue:&unary_fill_op type:MTLDataTypeShort atIndex:1200];
        [unary_fill_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f16_f16"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_expand"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_expand_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_expand_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_dsv4_indexer_score_one_direct_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexer_score_one_direct");
        g_dsv4_compressor_store_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_compressor_store_one");
        g_dsv4_sort_i32_rows_asc_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_sort_i32_rows_asc");
        g_dsv4_indexed_attention_heads8_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8");
        g_dsv4_indexed_attention_heads8_rb16_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb16");
        g_dsv4_softplus_sqrt_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_softplus_sqrt_f32_4");
        g_dsv4_router_finalize_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one");
        g_dsv4_router_weights_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_weights_one");
        g_dsv4_route_translate_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_route_translate");
        g_dsv4_hc_expand4_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_hc_expand4");
        if (!g_dsv4_indexer_score_one_direct_pipeline ||
            !g_dsv4_compressor_store_one_pipeline ||
            !g_dsv4_sort_i32_rows_asc_pipeline ||
            !g_dsv4_indexed_attention_heads8_pipeline ||
            !g_dsv4_indexed_attention_heads8_rb16_pipeline ||
            !g_dsv4_softplus_sqrt_pipeline ||
            !g_dsv4_router_finalize_one_pipeline ||
            !g_dsv4_router_weights_one_pipeline ||
            !g_dsv4_route_translate_pipeline ||
            !g_dsv4_hc_expand4_pipeline) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
    }

    return 1;
}
