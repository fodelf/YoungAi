/* metal_pipeline.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d",
                     function_name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

id<MTLComputePipelineState> ds4_gpu_get_mul_mm_id_pipeline(
        const char *function_name,
        bool        bc_inp) {
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d",
                     function_name, bc_inp ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

id<MTLComputePipelineState> ds4_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found\n", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

id<MTLComputePipelineState> ds4_gpu_hot_pipeline(
        id<MTLComputePipelineState> pipeline,
        const char *fallback_name) {
    (void)fallback_name;
    return pipeline;
}

/* Both gates must be set before ds4_gpu_init: strict-fp picks the shader
 * preprocessor macros and math mode at library compile time, and the Metal 4
 * gate feeds ds4_gpu_detect_metal4_features which runs during init. */
void ds4_gpu_set_strict_fp(int on) {
    g_strict_fp = on ? 1 : 0;
}

void ds4_gpu_set_metal4_enabled(int on) {
    g_metal4_enabled = on ? 1 : 0;
}

int ds4_gpu_mpp_available(void) {
    return g_metal4_tensor_api_enabled && !g_quality_mode;
}

/*
 * Retained Metal4 defaults live here instead of behind user-visible options.
 * The public runtime has one automatic accelerated path plus the global
 * ds4_gpu_set_metal4_enabled comparison switch.  Benchmark-only alternatives
 * that lost during M5 work are removed or kept out of the dispatch path so
 * future changes do not accidentally turn old experiments into new modes.
 */
int ds4_gpu_use_mpp_attn_out_low_matmul(void) {
    return ds4_gpu_mpp_available();
}

void ds4_gpu_warn_mpp_fallback(void) {
    static int warned;
    if (!warned) {
        fprintf(stderr, "ds4: accelerated Metal prefill matmul unavailable; falling back to legacy kernel\n");
        warned = 1;
    }
}

int ds4_gpu_device_name_contains(const char *needle) {
    return g_metal_device_name[0] != '\0' && strstr(g_metal_device_name, needle) != NULL;
}

const char *ds4_gpu_device_name(void) { return g_metal_device_name; }

static int ds4_gpu_compile_tensor_probe(void) {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (!g_device) return 0;
    if (@available(macOS 26.0, *)) {
        const char *src =
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void ds4_tensor_probe(\n"
            "        tensor<device half,  dextents<int32_t, 2>> A [[buffer(0)]],\n"
            "        tensor<device half,  dextents<int32_t, 2>> B [[buffer(1)]],\n"
            "        device float *C [[buffer(2)]],\n"
            "        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
            "    auto tA = A.slice(0, (int)tgid.y);\n"
            "    auto tB = B.slice((int)tgid.x, 0);\n"
            "    matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
            "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n"
            "    auto sA = tA.slice(0, 0);\n"
            "    auto sB = tB.slice(0, 0);\n"
            "    mm.run(sB, sA, cT);\n"
            "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16));\n"
            "    cT.store(tC);\n"
            "}\n";

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:src];
        id<MTLLibrary> probe_library = [g_device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!probe_library) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe compile failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        id<MTLFunction> fn = [probe_library newFunctionWithName:@"ds4_tensor_probe"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe function missing\n");
            return 0;
        }
        error = nil;
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            fprintf(stderr, "ds4: Metal 4 tensor API probe pipeline failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
        return 1;
    }
#endif
    return 0;
}

void ds4_gpu_detect_metal4_features(void) {
    g_metal4_runtime_available = 0;
    g_metal4_family_supported = 0;
    g_metal4_queue_supported = 0;
    g_metal4_m5_neural_accelerators_hint = 0;
    g_metal4_tensor_api_enabled = 0;
    g_metal4_tensor_api_compile_supported = 0;
    g_metal_device_name[0] = '\0';

    if (!g_device) return;

    const char *name = [[g_device name] UTF8String];
    if (name) {
        snprintf(g_metal_device_name, sizeof(g_metal_device_name), "%s", name);
    }

    const int metal4_disabled = !g_metal4_enabled;

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
        g_metal4_runtime_available = 1;
        g_metal4_family_supported =
            !metal4_disabled && [g_device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        g_metal4_queue_supported = [g_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;

        /*
         * Apple does not currently expose a separate "Neural Accelerator" bit
         * through Metal. On public M5 systems the hardware signal is the device
         * generation plus Metal 4 support, so keep this as a conservative hint.
         */
        if (g_metal4_family_supported && ds4_gpu_device_name_contains("M5")) {
            g_metal4_m5_neural_accelerators_hint = 1;
        }

        if (g_metal4_family_supported) {
            const int default_enable =
                ds4_gpu_device_name_contains("M5") ||
                ds4_gpu_device_name_contains("M6") ||
                ds4_gpu_device_name_contains("A19") ||
                ds4_gpu_device_name_contains("A20");

            /*
             * Metal 4 TensorOps are portable in source, but on pre-M5 hardware
             * they can map to ordinary shader fallbacks.  Keep the automatic
             * fast path restricted to hardware generations where the Neural
             * Accelerator/TensorOps path is expected to pay off; older Metal
             * machines continue to use the established kernels unless a future
             * device is explicitly added here.
             */
            if (default_enable) {
                g_metal4_tensor_api_compile_supported = ds4_gpu_compile_tensor_probe();
                g_metal4_tensor_api_enabled = g_metal4_tensor_api_compile_supported;
                if (!g_metal4_tensor_api_enabled) {
                    fprintf(stderr, "ds4: Metal 4 tensor API probe failed; using legacy Metal kernels\n");
                }
            } else {
                fprintf(stderr, "ds4: Metal 4 tensor API disabled for pre-M5/pre-A19 devices\n");
            }
        }
    }
#endif
}

int ds4_gpu_warm_model_views(void) {
    if (g_model_view_count == 0) return 1;

    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_touch_u8_stride");
    if (!pipeline) return 0;

    /* 1 MiB stride: a validation touch over the VM ranges, not a prefetch. */
    const uint64_t stride = 1024ull * 1024ull;

    uint64_t total_touches = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (!g_model_views[i].resident_hint) continue;
        total_touches += (g_model_views[i].bytes + stride - 1) / stride;
    }
    if (total_touches == 0) return 1;
    if (total_touches > (uint64_t)NSUIntegerMax) return 0;

    const NSUInteger out_bytes = (NSUInteger)total_touches;
    id<MTLBuffer> out = [g_device newBufferWithLength:out_bytes
                                             options:MTLResourceStorageModeShared];
    if (!out) {
        fprintf(stderr, "ds4: Metal model warmup scratch allocation failed\n");
        return 0;
    }
    out.label = @"ds4_model_warmup";

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) {
        fprintf(stderr, "ds4: Metal model warmup command buffer allocation failed\n");
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    uint64_t dst_offset = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (!g_model_views[i].resident_hint) continue;
        const uint64_t bytes = g_model_views[i].bytes;
        const uint64_t n = (bytes + stride - 1) / stride;
        [enc setBuffer:g_model_views[i].buffer offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
        [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dst_offset += n;
    }
    ds4_gpu_end_compute_encoder(cb, enc);

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal model warmup failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return 1;
}

const char *ds4_gpu_mul_mm_id_map0_name(uint32_t ne20) {
    switch (ne20) {
        case 1:  return "kernel_mul_mm_id_map0_ne20_1";
        case 2:  return "kernel_mul_mm_id_map0_ne20_2";
        case 4:  return "kernel_mul_mm_id_map0_ne20_4";
        case 5:  return "kernel_mul_mm_id_map0_ne20_5";
        case 6:  return "kernel_mul_mm_id_map0_ne20_6";
        case 8:  return "kernel_mul_mm_id_map0_ne20_8";
        case 10: return "kernel_mul_mm_id_map0_ne20_10";
        case 16: return "kernel_mul_mm_id_map0_ne20_16";
        case 22: return "kernel_mul_mm_id_map0_ne20_22";
        default: return NULL;
    }
}
