/* t_metal_v41_group.c — V4.1 Metal 原语回归的组入口(2026-10-08)。四段各住一个文件(守 500 行): 稠密 / 层内+注意力 / VQ 专家 / 反传有限差分。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
void test_metal_v41_group(void) {
    test_metal_v41_dense();
    test_metal_v41_layer();
    test_metal_v41_attn();
    test_metal_v41_vq();
    test_metal_v41_bwd();
}
#else
typedef int ds4_t_metal_v41_group_nonempty_tu;
#endif
