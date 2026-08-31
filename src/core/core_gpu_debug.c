/* core_gpu_debug.c — ffn_out 物化辅助 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    (void)il;
    (void)pos;
    return metal_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out;
}

bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}

bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_debug_nonempty_tu; /* 空TU防御(CPU构建) */
