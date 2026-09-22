/* v41_route_solve.h — 路由偏置侧车(路由反修)的 GPU 数值件(2026-09-20)。驱动在 v41_rb_run.inc.c。
 *
 * 路由重算与引擎 v41_router_kernel(src/cuda/cuda_v41_3.inc.cu)逐式同:
 *   logits = x·Wgᵀ(bf16 权重, f32 累加); p = sqrt(softplus(z))(softplus 阈 20, 与 torch 同); score = p + bias(+Δb);
 *   top-K 按 score, 同分取小号; w = p / (Σ_{选中} p + 1e-20) · route_scale。
 * 为什么在工具里重算而不从引擎要: 钩子只给选中的 6 个(sel/rw), 而 margin 缺口要全部 384 个专家的选择分;
 * 给引擎加取料口就是改引擎(用户令: 引擎不为反修加旗标)。重算后与钩子的 sel 对表是硬闸(见驱动)。 */
#ifndef V41_ROUTE_SOLVE_H
#define V41_ROUTE_SOLVE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* 路由权重上设备: bf16 [E][D] → f32; bias f32 [E]。调用方用完 cudaFree 两个指针。返回 0 成功。 */
int v41_route_upload_gate_gpu(const uint16_t *hWg_bf16, const float *hB, int E, int D, float **dWg_out, float **dB_out);

/* 路由重算(全设备指针)。dDb 非 NULL = 选择分再加 Δb[E]; dscore_out [n][E] 必须给(选择分, 含 bias 与 Δb)。返回 0 成功。 */
int v41_route_recompute_gpu(const float *dX, const float *dWg, const float *dB, const float *dDb, int n, int D, int E, int K, float scale,
                            int *dsel_out, float *dw_out, float *dscore_out);

#ifdef __cplusplus
}
#endif
#endif
