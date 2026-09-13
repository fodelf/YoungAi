/* v41_fp_moe.h — 反修靶用的 FP(出厂精度)MoE 块评估器接口(CUDA 实现在 v41_fp_moe.cu; 2026-09-13)。
 *
 * 【干什么】靶 = y_fp(x_q) − y_q(x_q): 同一个 x(引擎钩子给的部署态 MoE 输入), 只把权重换成 HF 出厂件
 * (routed 专家 FP4 E2M1 1×32 ue8m0 / shared 专家 E4M3 32×32 块 ue8m0), 算一遍 MoE 块出口。
 * 【舍入点逐式照引擎 CUDA 路】(cuda_vq_prefill.inc.cu / cuda_v41_1,3.inc.cu): routed 激活 f32→f16, gate|up 合一
 * f16 GEMM f32 累加 → SwiGLU(截断)→f16 → down GEMM f32 → 按 pick 序定序加权 reduce; shared 激活 bf16→f16,
 * 每段线性出口舍 bf16, h 舍 bf16; y = bf16(routed + shared)。这样 y_fp 与 y_q 只差权重, 靶不混进实现差。 */
#ifndef V41_FP_MOE_H
#define V41_FP_MOE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t *w, *s;   /* 主机指针(HF 分片 mmap): 权重字节 / ue8m0 scale 字节 */
    int rows, cols;         /* 逻辑形状 [rows][cols](FP4 打包文件里存的是 cols/2 字节) */
    int sbr, sbc;           /* scale 块: FP4 = 1×32; E4M3 = 32×32(由 scale 张量形状算出) */
    int fp8;                /* 1 = E4M3 + 块 scale(shared); 0 = FP4 E2M1 打包 + 每行每 32 列 scale(routed) */
} v41_fp_mat;
typedef struct { v41_fp_mat w1, w3, w2; } v41_fp_ffn;   /* w1/w3 [MID][D], w2 [D][MID] */

/* dY[n][D](设备 f32) = 出厂权重 MoE(dX[n][D] 设备 f32): routed 按 hsel/hrw(主机 [n][n_used], 引擎路由原样) + shared,
 * 出口 bf16 格点。clamp = SwiGLU 截断(引擎 DS4_SWIGLU_CLAMP_EXP)。返回 0 成功, 非 0 = CUDA/形状错(调用方停车)。 */
int v41_fp_moe_layer(const v41_fp_ffn *experts, int n_expert, const v41_fp_ffn *shared,
                     const float *dX, const int *hsel, const float *hrw, int n, int D, int MID, int n_used, float clamp,
                     float *dY);

#ifdef __cplusplus
}
#endif
#endif
