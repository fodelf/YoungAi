/* metal_moe_thin.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。
 * 历史: 本文件曾装着 MoE 降激活(thinning)/keep-sim/expert-half overlap 三套
 * env 独占实验设施, 全部删除 —— 数量裁专家是在案质量灾难(2026-07-06: topk=4
 * 把逐字正确的贪心 Go 打成词汤), overlap 是从未默认开的 A/B 结构。只留
 * grouped-GEMM 阈值。 */
#import "metal_internal.h"

/* Minimum batch token count for the grouped-GEMM (mm_id) routed-MoE path.
 * mm_id maps token-rows per expert and reads each active expert's weights
 * exactly ONCE (expert-major), while the mv/pair paths re-read the expert
 * weights once per (expert,token) pair; the 5..12-token verify shapes sit
 * right in the slow gap of the old >=32 split (measured 2026-06-10). */
uint32_t ds4_gpu_moe_mm_id_min(void) {
    return DS4_METAL_MOE_MM_ID_MIN_TOKENS;
}
