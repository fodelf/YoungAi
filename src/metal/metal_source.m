/* metal_source.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

void ds4_gpu_print_memory_report(const char *label) {
    const uint64_t scratch =
        (uint64_t)g_flash_attn_mask_bytes +
        (uint64_t)g_flash_attn_pad_bytes +
        (uint64_t)g_flash_attn_tmp_bytes +
        (uint64_t)g_flash_attn_blk_bytes +
        (uint64_t)g_flash_attn_ring_bytes +
        (uint64_t)g_flash_attn_kv_bytes +
        (uint64_t)g_compressor_pool_kv_bytes +
        (uint64_t)g_compressor_pool_score_bytes +
        (uint64_t)g_compressor_pool_score_cont_bytes +
        (uint64_t)g_compressor_pool_softmax_bytes +
        (uint64_t)g_compressor_pool_product_bytes +
        (uint64_t)g_compressor_store_ape_bytes +
        (uint64_t)g_compressor_store_score_bytes +
        (uint64_t)g_embed_rows_bytes +
        (uint64_t)g_router_selection_bytes +
        (uint64_t)g_router_weight_sum_bytes +
        (uint64_t)g_indexer_head_scores_bytes +
        (uint64_t)g_indexer_topk_bytes +
        (uint64_t)g_indexed_topk_bytes +
        (uint64_t)g_f16_round_scratch_bytes +
        (uint64_t)g_raw_store_round_bytes +
        (uint64_t)g_moe_gate_scratch_bytes +
        (uint64_t)g_moe_down_scratch_bytes +
        (uint64_t)g_moe_id_map_bytes;

    fprintf(stderr, "ds4: Metal memory report%s%s\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "");
    fprintf(stderr,
            "ds4:   runtime tensors live %.2f MiB peak %.2f MiB\n",
            ds4_gpu_mib(g_tensor_alloc_live_bytes),
            ds4_gpu_mib(g_tensor_alloc_peak_bytes));
    fprintf(stderr,
            "ds4:   mmap model wrapper spans %llu buffers %.2f GiB total, %.2f GiB max (not copied)\n",
            (unsigned long long)g_model_wrap_count,
            ds4_gpu_gib(g_model_wrap_bytes),
            ds4_gpu_gib(g_model_wrap_max_bytes));
    fprintf(stderr,
            "ds4:   model residency requests %llu\n",
            (unsigned long long)g_model_residency_count);
    fprintf(stderr,
            "ds4:   device %s, Metal 4 runtime %s, family %s, MTL4 queue %s, tensor API %s, M5 neural accelerators %s\n",
            g_metal_device_name[0] ? g_metal_device_name : "(unknown)",
            g_metal4_runtime_available ? "yes" : "no",
            g_metal4_family_supported ? "yes" : "no",
            g_metal4_queue_supported ? "yes" : "no",
            g_metal4_tensor_api_enabled ? "enabled" :
                (g_metal4_tensor_api_compile_supported ? "available" : "disabled"),
            g_metal4_m5_neural_accelerators_hint ? "likely" : "not detected");
    fprintf(stderr,
            "ds4:   accelerated Metal path %s%s\n",
            ds4_gpu_mpp_available() ? "enabled" : "disabled",
            g_quality_mode ? " by --quality" :
                (!g_metal4_tensor_api_enabled ? " (tensor API unavailable)" : ""));
    fprintf(stderr,
            "ds4:   scratch %.2f MiB (flash mask %.2f, pad %.2f, tmp %.2f, blk %.2f, ring %.2f, kv %.2f, compressor %.2f, router %.2f, indexer %.2f, moe %.2f, f16 %.2f, raw-store %.2f)\n",
            ds4_gpu_mib(scratch),
            ds4_gpu_mib((uint64_t)g_flash_attn_mask_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_pad_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_tmp_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_blk_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_ring_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_kv_bytes),
            ds4_gpu_mib((uint64_t)g_compressor_pool_kv_bytes +
                          (uint64_t)g_compressor_pool_score_bytes +
                          (uint64_t)g_compressor_pool_score_cont_bytes +
                          (uint64_t)g_compressor_pool_softmax_bytes +
                          (uint64_t)g_compressor_pool_product_bytes +
                          (uint64_t)g_compressor_store_ape_bytes +
                          (uint64_t)g_compressor_store_score_bytes +
                          (uint64_t)g_embed_rows_bytes),
            ds4_gpu_mib((uint64_t)g_router_selection_bytes +
                          (uint64_t)g_router_weight_sum_bytes),
            ds4_gpu_mib((uint64_t)g_indexer_head_scores_bytes +
                          (uint64_t)g_indexer_topk_bytes +
                          (uint64_t)g_indexed_topk_bytes),
            ds4_gpu_mib((uint64_t)g_moe_gate_scratch_bytes +
                          (uint64_t)g_moe_down_scratch_bytes +
                          (uint64_t)g_moe_id_map_bytes),
            ds4_gpu_mib((uint64_t)g_f16_round_scratch_bytes),
            ds4_gpu_mib((uint64_t)g_raw_store_round_bytes));
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

static const char *ds4_gpu_source =
"#include <metal_stdlib>\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef DS4_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"\n";

NSString *ds4_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:ds4_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  超 500 行的原文件按保序方案拆成组, 逐文件按序拼接的结果与
     * 拆分前逐字节一致 —— 文件序不能动, 拼接顺序决定 Metal library 内容。
     */
    NSArray<NSString *> *required_sources = @[
        @"metal/flash_attn_pad.metal",
        @"metal/flash_attn_ext.metal",
        @"metal/flash_attn_vec.metal",
        @"metal/dense_mv.metal",
        @"metal/dense_mv_ext.metal",
        @"metal/dense_mm_mpp.metal",
        @"metal/dense_mm.metal",
        @"metal/moe_core.metal",
        @"metal/moe_dequant.metal",
        @"metal/moe_mv_impl.metal",
        @"metal/moe_mv_id.metal",
        @"metal/moe_mv_pair.metal",
        @"metal/moe_mm_id.metal",
        @"metal/moe_sidecar.metal",
        @"metal/dsv4_hc_sum.metal",
        @"metal/dsv4_hc_expand.metal",
        @"metal/unary.metal",
        @"metal/dsv4_kv.metal",
        @"metal/dsv4_rope.metal",
        @"metal/dsv4_misc_router.metal",
        @"metal/dsv4_misc_attend.metal",
        @"metal/dsv4_misc_indexer.metal",
        @"metal/dsv4_misc_pool.metal",
        @"metal/argsort.metal",
        @"metal/cpy.metal",
        @"metal/concat.metal",
        @"metal/get_rows.metal",
        @"metal/sum_rows.metal",
        @"metal/softmax.metal",
        @"metal/repeat.metal",
        @"metal/glu.metal",
        @"metal/norm.metal",
        @"metal/bin.metal",
        @"metal/set_rows.metal",
        /* DeepSeek V4.1 一族(2026-10-08): 公共件必须在最前(后面的文件用它的函数与参数块), 其余按依赖序 */
        @"metal/v41_common.metal",
        @"metal/v41_dense.metal",
        @"metal/v41_layer.metal",
        @"metal/v41_attn.metal",
        @"metal/v41_sample.metal",
        @"metal/v41_vq.metal",
        @"metal/v41_bwd.metal",
        @"metal/v41_draft.metal",
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSString *entry in required_sources) {
        NSArray<NSString *> *paths =
            @[entry, [@"./" stringByAppendingString:entry]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "ds4: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            fprintf(stderr, "ds4: Metal source %s not found\n", [entry UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}
