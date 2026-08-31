/* metal_expert_profile.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

/* host AUTO verdict: -1 unset, else 0/1 */
static int g_expert_offload_cached  = -1;

void ds4_gpu_set_expert_offload(int enabled) {
    g_expert_offload_verdict = enabled ? 1 : 0;
    g_expert_offload_cached = -1;   /* re-resolve on next query with the new verdict */
}

uint64_t ds4_gpu_recommended_max_working_set_bytes(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_device == nil) return 0;
    return (uint64_t)[g_device recommendedMaxWorkingSetSize];
}

/* Live GPU working-set size (model wired + graph scratch). Compared against
 * recommendedMaxWorkingSetSize this is the absolute evidence for whether a
 * slice fits in VRAM without paging. */
uint64_t ds4_gpu_current_allocated_bytes(void) {
    if (g_device == nil) return 0;
    return (uint64_t)[g_device currentAllocatedSize];
}

int ds4_gpu_expert_offload_enabled(void) {
    if (g_expert_offload_cached < 0) {
        const int decision = (g_expert_offload_verdict >= 0) ? g_expert_offload_verdict : 0;
        g_expert_offload_cached = decision;
        if (decision) {
            fprintf(stderr,
                    "ds4: routed-expert offload ON (mmap views non-resident, per-layer CPU "
                    "gather; over-budget model verdict).\n");
        } else {
            fprintf(stderr,
                    "ds4: routed experts RESIDENT (wired, direct GPU read, no per-layer gather; "
                    "model fits budget).\n");
        }
    }
    return g_expert_offload_cached;
}
